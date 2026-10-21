---
status: accepted
---

# Updates follow Release tags from the checkout's own upstream

The repo lives on several remotes (GitLab KIT as `origin`, public GitHub, a private GitHub mirror), and users install by cloning one of them. Instead of hardcoding one update source, an install looks for newer Release tags (`vX.Y.Z`) on the remote its own checkout came from. The app only checks and offers the Update. The launcher (`VoxScribe.bat`, the macOS `.app` script) carries it out on the next start: it checks out the new tag, runs `uv sync`, and goes back to the previous tag if that fails.

## Considered Options

- **Follow `main`.** Rejected: every push would reach users immediately, including unfinished work.
- **One hardcoded remote.** Rejected: GitLab KIT needs credentials that external users lack, and the public GitHub repo lags behind for internal users. Each user can already reach the remote they cloned from.
- **The app updates itself while running.** Rejected: it would replace files it is running from, and could interrupt a recording or transcription.

## Consequences

- Only a clean checkout sitting exactly on a Release tag (detached HEAD) is updated. Branches, local edits and commits past the tag are skipped silently, so development checkouts are never touched.
- Releases must be pushed to every remote users clone from, together with their tag. The release script does this, and the remotes must not drift apart.
- A failed check (offline, no VPN, no GitLab credentials) is only logged, never shown as an error.
