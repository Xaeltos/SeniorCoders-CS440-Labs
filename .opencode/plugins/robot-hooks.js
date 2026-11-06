
import { execFile } from "node:child_process";
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

const TERMINAL_EVENTS = new Set([
    "session.execution.succeeded",
    "session.execution.failed",
    "session.execution.interrupted"
]);

// Kept intentionally narrow. Subscribing to every event (especially
// filesystem.changed) would make our own log writes re-trigger this plugin.
const SUBSCRIBED_EVENTS = [
    ...TERMINAL_EVENTS,
    "session.status",
    "session.idle",
    "session.tool.success",
    "session.tool.failed"
];

export default {
    id: "robot-hooks",

    setup(ctx) {
        const directory = ctx.location.directory;
        const openCodeDir = join(directory, ".opencode");

        const hookTest = join(openCodeDir, "hook-test.txt");
        const eventsLog = join(openCodeDir, "events.log");
        const resultsLog = join(openCodeDir, "hook-results.log");

        const script = join(
            openCodeDir,
            "scripts",
            "validate_robot.py"
        );

        mkdirSync(dirname(eventsLog), { recursive: true });

        // Confirm successful plugin initialization
        writeFileSync(
            hookTest,
            "OpenCode V2 plugin initialized successfully"
        );

        const controller = new AbortController();

        // Re-entrancy and duplicate guards. The validation process itself
        // never starts an OpenCode session, but these guards also protect
        // against bursty or repeated terminal events for the same turn.
        let running = false;
        let lastStartedAt = 0;
        let child = null;

        const seenEventIDs = new Set();
        const seenOrder = [];
        const COOLDOWN_MS = 1500;
        const MAX_SEEN = 200;

        const append = (file, text) => {
            try {
                appendFileSync(file, text);
            } catch {
                // Logging must never take down the server or event stream.
            }
        };

        const logEvent = (event) => {
            const data = event?.data ?? {};
            const summary = {
                type: event?.type,
                id: event?.id,
                sessionID: data.sessionID,
                status: data.status?.type ?? data.status,
                tool: data.tool?.name ?? data.name,
                callID: data.callID ?? data.id
            };

            append(
                eventsLog,
                `[${new Date().toISOString()}] ${JSON.stringify(summary)}\n`
            );
        };

        const runValidation = (event) => {
            // Hard recursion guard: a validation child never inherits a
            // state where it is allowed to trigger another hook run.
            if (process.env.ROBOT_HOOK_ACTIVE === "1") {
                return;
            }

            const eventID = event?.id;
            if (eventID && seenEventIDs.has(eventID)) {
                return;
            }

            const now = Date.now();
            if (running || now - lastStartedAt < COOLDOWN_MS) {
                return;
            }

            running = true;
            lastStartedAt = now;

            if (eventID) {
                seenEventIDs.add(eventID);
                seenOrder.push(eventID);
                if (seenOrder.length > MAX_SEEN) {
                    seenEventIDs.delete(seenOrder.shift());
                }
            }

            child = execFile(
                "python",
                [script],
                {
                    cwd: directory,
                    encoding: "utf8",
                    timeout: 30000,
                    // Marker available to any nested process. It prevents a
                    // child that somehow reaches OpenCode from re-firing this.
                    env: { ...process.env, ROBOT_HOOK_ACTIVE: "1" }
                },
                (error, stdout, stderr) => {
                    running = false;
                    child = null;

                    if (error) {
                        append(
                            resultsLog,
                            `[${new Date().toISOString()}]\n` +
                            `HOOK FAILED\n` +
                            `${stdout || ""}\n` +
                            `${stderr || error.message}\n`
                        );
                        return;
                    }

                    append(
                        resultsLog,
                        `[${new Date().toISOString()}]\n` +
                        `HOOK PASSED\n${stdout}\n`
                    );
                }
            );
        };

        void (async () => {
            try {
                for await (const event of ctx.event.subscribe({
                    types: SUBSCRIBED_EVENTS,
                    signal: controller.signal
                })) {
                    logEvent(event);

                    // session.execution.* is the terminal lifecycle event for
                    // one complete agent execution (prompt -> response/tools).
                    if (TERMINAL_EVENTS.has(event.type)) {
                        runValidation(event);
                    }
                }
            } catch (error) {
                if (!controller.signal.aborted) {
                    append(
                        resultsLog,
                        `[${new Date().toISOString()}]\n` +
                        `EVENT SUBSCRIPTION FAILED\n` +
                        `${error?.stack || error?.message || String(error)}\n`
                    );
                }
            }
        })();

        return () => {
            controller.abort();
            if (child) {
                child.kill();
                child = null;
            }
        };
    }
};
