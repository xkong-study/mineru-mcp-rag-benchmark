# Engineering Playbook

## Workflow

1. Open a branch for your change.
2. Keep pull requests small and reviewable.
3. Prefer clear, testable logic over hidden behavior.
4. Document operational changes when a workflow changes.

## Deployment and verification

- To release a feature safely, keep the change small, verify the target environment, and check the expected user-facing state before and after launch.
- Verify the target environment before launch.
- Check logs and telemetry after rollout.
- If a change touches user-facing state, validate the edge cases.
- Record rollback notes when the change affects production behavior.

## What new joiners should learn first

- How the code review process works.
- Where deployment notes are stored.
- Who owns frontend, backend, and platform workflows.
- What counts as a production incident.

## Useful questions

- How do I release a feature safely?
- Where are engineering runbooks stored?
- What is the rollback process?
