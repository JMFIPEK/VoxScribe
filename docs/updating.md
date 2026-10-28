# Updating

## From v1.1.0 on: in the app

When a newer Release is published, VoxScribe shows a notice after it starts:

- **Install and relaunch**: VoxScribe quits, installs the Update and starts again.
- **Install and quit**: the same, but it doesn't start again afterwards.

This works on Windows (started via the shortcut / `VoxScribe.bat`) and on
macOS (started via VoxScribe.app; see [macOS setup](macos.md) for when to
re-run `macos/make-app.sh`). Started any other way, the Update is installed
the next time you start VoxScribe from the shortcut or the app.

Both buttons are disabled while you record or transcribe. If the Update fails,
the previous Release is restored, and the app tells you so on its next start
(details in `~/.voxscribe/voxscribe.log`). Settings → Info shows the installed
Release, e.g. `v1.1.1`.

## Installs on `main` or older than v1.1.0: once by hand

Installs cloned before Releases existed sit on the `main` branch, and
Settings → Info shows them as "development". Move onto the newest Release once
(commit or stash local edits first), in the VoxScribe folder:

```bash
git fetch --tags origin
```

```bash
git checkout $(git describe --tags --abbrev=0 --match "v*" origin/main)
```

Then start VoxScribe as usual: the launcher syncs the environment because
`uv.lock` changed. Use the remote you cloned from in place of `origin` if it's
named differently (`git remote -v`). From then on, Updates come through the
app.

"development" is also shown for a checkout with local edits to tracked files,
on a branch, or with commits past the Release tag. Such installs are not
offered Updates.
