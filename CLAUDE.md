# Project: LSP

## Language & Tools
- Rust (edition 2024)
- cargo, clippy, rustfmt
- Tests: cargo test / cargo nextest

## Code Standards
- Run `cargo fmt` before commit
- No clippy warnings allowed
- All public APIs must have doc comments
- Error handling with thiserror/anyhow, no unwrap() in production code
- Write unit tests for all new logic

## Git Conventions
- Conventional commits: feat:, fix:, refactor:, test:, docs:, chore:
- One commit per logical change

## Teamwork
- Architecture changes require plan approval
- Code review required before merge
- All tests must pass before a task is marked complete
