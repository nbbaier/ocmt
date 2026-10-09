/**
 * OpenCode AI client for generating commit messages and changelogs
 *
 * Talks to the opencode v2 server HTTP API (/api/*, Basic auth)
 */

import { type ChildProcess, exec, spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { promisify } from "node:util";
import * as p from "@clack/prompts";
import color from "picocolors";
import { getChangelogConfig, getCommitConfig } from "./config";

const execAsync = promisify(exec);

// The v2 server checks the password and always expects this username
const SERVER_USERNAME = "opencode";
const DEFAULT_SERVER_URL = "http://127.0.0.1:4096";
const SERVER_START_TIMEOUT = 10000;

interface ModelRef {
	providerID: string;
	id: string;
}

// Models
const COMMIT_MODEL: ModelRef = {
	providerID: "opencode",
	id: "gpt-5-nano",
};

const CHANGELOG_MODEL: ModelRef = {
	providerID: "opencode",
	id: "claude-sonnet-4-5",
};

interface Connection {
	baseUrl: string;
	authorization: string;
}

// Server state
let connectionInstance: Connection | null = null;
let serverProcess: ChildProcess | null = null;

export interface CommitGenerationOptions {
	diff: string;
	context?: string;
}

export interface ChangelogGenerationOptions {
	commits: Array<{ hash: string; message: string }>;
	diff?: string;
	fromRef: string;
	toRef: string;
	version?: string | null;
}

export interface UpdateChangelogOptions {
	newChangelog: string;
	existingChangelog: string;
	changelogPath: string;
}

/**
 * Check if opencode CLI is installed
 */
async function isOpencodeInstalled(): Promise<boolean> {
	try {
		await execAsync("which opencode");
		return true;
	} catch {
		return false;
	}
}

function createConnection(baseUrl: string, password: string): Connection {
	const credentials = Buffer.from(`${SERVER_USERNAME}:${password}`).toString(
		"base64",
	);
	return {
		baseUrl: baseUrl.replace(/\/$/, ""),
		authorization: `Basic ${credentials}`,
	};
}

/**
 * Send a JSON request to the OpenCode server
 */
async function request<T>(
	connection: Connection,
	method: string,
	path: string,
	body?: unknown,
): Promise<T> {
	const response = await fetch(`${connection.baseUrl}${path}`, {
		method,
		headers: {
			authorization: connection.authorization,
			...(body === undefined ? {} : { "content-type": "application/json" }),
		},
		body: body === undefined ? undefined : JSON.stringify(body),
	});

	if (!response.ok) {
		const text = await response.text();
		let message = text;
		try {
			message = JSON.parse(text).message ?? text;
		} catch {
			// Not JSON, use the raw body
		}
		throw new Error(
			`OpenCode ${method} ${path} failed (${response.status}): ${message}`,
		);
	}

	if (response.status === 204) {
		return undefined as T;
	}

	// v1 servers and the web UI answer unknown routes with HTML
	if (!response.headers.get("content-type")?.includes("application/json")) {
		throw new Error(
			`OpenCode ${method} ${path} returned non-JSON; is the server opencode v2?`,
		);
	}

	return (await response.json()) as T;
}

/**
 * Connect to an already running server, if we have its password
 */
async function connectExisting(): Promise<Connection | null> {
	const password =
		process.env.OPENCODE_PASSWORD ?? process.env.OPENCODE_SERVER_PASSWORD;
	if (!password) {
		return null;
	}

	const connection = createConnection(
		process.env.OPENCODE_SERVER_URL ?? DEFAULT_SERVER_URL,
		password,
	);
	try {
		await request(connection, "GET", "/api/provider");
		return connection;
	} catch {
		return null;
	}
}

/**
 * Spawn `opencode serve` with a generated password and wait until it listens
 */
function spawnServer(): Promise<Connection> {
	const password = randomBytes(32).toString("base64url");
	// OPENCODE_PASSWORD takes precedence over OPENCODE_SERVER_PASSWORD
	const { OPENCODE_PASSWORD: _, ...env } = process.env;

	const proc = spawn("opencode", ["serve", "--hostname=127.0.0.1"], {
		env: { ...env, OPENCODE_SERVER_PASSWORD: password },
	});
	serverProcess = proc;

	return new Promise((resolve, reject) => {
		let output = "";

		const fail = (error: Error) => {
			clearTimeout(timer);
			proc.kill();
			reject(error);
		};

		const timer = setTimeout(
			() =>
				fail(
					new Error(
						`Timeout waiting for server to start after ${SERVER_START_TIMEOUT}ms`,
					),
				),
			SERVER_START_TIMEOUT,
		);

		proc.stdout?.on("data", (chunk) => {
			output += chunk.toString();
			const match = output.match(/server listening on (https?:\/\/\S+)/);
			if (match) {
				clearTimeout(timer);
				resolve(createConnection(match[1], password));
			}
		});
		proc.stderr?.on("data", (chunk) => {
			output += chunk.toString();
		});
		proc.on("error", fail);
		proc.on("exit", (code) => {
			clearTimeout(timer);
			reject(
				new Error(
					`Server exited with code ${code}${output.trim() ? `\nServer output: ${output}` : ""}`,
				),
			);
		});
	});
}

/**
 * Get or create the OpenCode server connection
 * Tries to connect to existing server first, spawns new one if needed
 */
async function getConnection(): Promise<Connection> {
	if (connectionInstance) {
		return connectionInstance;
	}

	// Try connecting to existing server first
	const existing = await connectExisting();
	if (existing) {
		connectionInstance = existing;
		return existing;
	}

	// Check if opencode is installed
	if (!(await isOpencodeInstalled())) {
		p.log.error("OpenCode CLI is not installed");
		p.log.info(
			`Install it with: ${color.cyan("npm install -g opencode")} or ${color.cyan("brew install sst/tap/opencode")}`,
		);
		process.exit(1);
	}

	// Spawn new server
	try {
		connectionInstance = await spawnServer();

		// Clean up server on process exit
		process.on("exit", () => {
			serverProcess?.kill();
		});
		process.on("SIGINT", () => {
			serverProcess?.kill();
			process.exit(0);
		});
		process.on("SIGTERM", () => {
			serverProcess?.kill();
			process.exit(0);
		});

		return connectionInstance;
	} catch (error: unknown) {
		if (error instanceof Error) {
			p.log.error(`Failed to start OpenCode server: ${error.message}`);
		} else {
			p.log.error(`Failed to start OpenCode server: ${String(error)}`);
		}
		p.log.info(`Make sure OpenCode is installed and configured correctly`);
		process.exit(1);
	}
}

/**
 * Run a one-off prompt in a temporary session and return the generated text
 */
async function generate(
	title: string,
	model: ModelRef,
	prompt: string,
): Promise<string> {
	const connection = await getConnection();

	const session = await request<{ data: { id: string } }>(
		connection,
		"POST",
		"/api/session",
		{ title, model, location: { directory: process.cwd() } },
	);
	const sessionPath = `/api/session/${encodeURIComponent(session.data.id)}`;

	try {
		const result = await request<{ data: { text: string } }>(
			connection,
			"POST",
			`${sessionPath}/generate`,
			{ prompt },
		);
		return result.data.text.trim();
	} finally {
		// Clean up session
		await request(connection, "DELETE", sessionPath).catch(() => {});
	}
}

/**
 * Generate a commit message from a git diff using OpenCode AI
 */
export async function generateCommitMessage(
	options: CommitGenerationOptions,
): Promise<string> {
	const { diff, context } = options;

	const systemPrompt = await getCommitConfig();

	// Build the prompt
	let prompt = `${systemPrompt}\n\n---\n\nGenerate a commit message for the following diff:\n\n\`\`\`diff\n${diff}\n\`\`\``;

	if (context) {
		prompt += `\n\nAdditional context: ${context}`;
	}

	const message = await generate("oc-commit", COMMIT_MODEL, prompt);

	if (!message) {
		throw new Error("No commit message generated");
	}

	// Clean up the message (remove markdown code blocks if present)
	return message
		.replace(/^```[\s\S]*?\n/, "")
		.replace(/\n```$/, "")
		.trim();
}

/**
 * Generate a changelog from commits using OpenCode AI
 */
export async function generateChangelog(
	options: ChangelogGenerationOptions,
): Promise<string> {
	const { commits, fromRef, toRef, version } = options;

	const systemPrompt = await getChangelogConfig();

	// Build the commits list
	const commitsList = commits
		.map((c) => `- ${c.hash}: ${c.message}`)
		.join("\n");

	// Build version instruction
	let versionInstruction = "";
	if (version) {
		versionInstruction = `\n\nIMPORTANT: A version bump to ${version} was detected. Use "[${version}]" as the version header with today's date (format: YYYY-MM-DD), NOT "[Unreleased]".`;
	} else {
		versionInstruction = `\n\nUse "[Unreleased]" as the version header since no version bump was detected.`;
	}

	// Build the prompt
	const prompt = `${systemPrompt}\n\n---\n\nGenerate a changelog for the following commits (from ${fromRef} to ${toRef}):${versionInstruction}\n\n${commitsList}`;

	const changelog = await generate("oc-changelog", CHANGELOG_MODEL, prompt);

	if (!changelog) {
		throw new Error("No changelog generated");
	}

	return changelog;
}

/**
 * Update an existing CHANGELOG.md file intelligently using AI
 * The AI will merge the new changelog content with existing content properly
 */
export async function updateChangelogFile(
	options: UpdateChangelogOptions,
): Promise<string> {
	const { newChangelog, existingChangelog } = options;

	const prompt = `You are updating a CHANGELOG.md file. Your task is to intelligently merge new changelog entries into the existing file.

## Rules:
1. Preserve the existing file structure and header
2. Add the new changelog entry in the correct position (newest entries at the top, after the header)
3. Do not duplicate entries - if similar entries exist, keep the most detailed version
4. Maintain consistent formatting with the existing file
5. Keep the "Keep a Changelog" format if that's what the file uses
6. If there's an existing [Unreleased] section, merge into it or replace it with the new content
7. Return ONLY the complete updated file content, no explanations

## Existing CHANGELOG.md:
\`\`\`markdown
${existingChangelog}
\`\`\`

## New changelog entry to add:
\`\`\`markdown
${newChangelog}
\`\`\`

Return the complete updated CHANGELOG.md content:`;

	let updatedChangelog = await generate(
		"oc-changelog-update",
		CHANGELOG_MODEL,
		prompt,
	);

	if (!updatedChangelog) {
		throw new Error("No updated changelog generated");
	}

	// Clean up markdown code blocks if present
	updatedChangelog = updatedChangelog
		.replace(/^```markdown\n?/i, "")
		.replace(/^```\n?/, "")
		.replace(/\n?```$/, "")
		.trim();

	return updatedChangelog;
}

/**
 * Cleanup function to close the server if we spawned one
 */
export function cleanup(): void {
	serverProcess?.kill();
	serverProcess = null;
	connectionInstance = null;
}
