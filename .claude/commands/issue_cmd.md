Please analyze and fix the Github issue: $ARGUMENTS.

Follow these steps:

# PLAN

## 1. Fetch the Issue

- Run `gh issue view <issue-number>` to retrieve title and description.
- Parse and store the title and summary into memory.

## 2. Analyze the Issue

- Understand the problem statement.
- Identify what is being asked (bug, feature, refactor, etc.).

## 3. Check Context and Prior Work

- Search `project_context.md` for relevant background or architectural notes.
- Search `claude.md` for any previous scratchpad/thoughts related to this topic.
- Search `current_sprint.md` to check if this issue aligns with sprint goals.
- Search GitHub PRs (`gh pr list` or relevant keyword queries) for history.
- Search the codebase for related files/functions.

## 4. Clarify Uncertainties

- If unclear, generate clarifying questions.
- Store these in `claude.md` or request clarification from user.

## 5. Plan the Work

- Ultrathink and Analyze, and Break down the problem into small, actionable steps.
- Identify inputs, outputs, edge cases, and data dependencies.
- Consider which tools or workflows (e.g. LangGraph, MCP) will be involved.

## 6. Create a New Scratchpad

- Create a new file under `scratchpads/` using this format:  
  `scratchpads/<issue-slug>.md`
- In that file, include:
  - A link to the GitHub issue.
  - Summary of the problem.
  - The action plan (bullet-pointed task breakdown).
  - Any relevant code links, notes from previous scratchpads, or dependencies.

---

# CREATE

## 1. Create a new Git branch for the issue

- Use the format: `issue/<issue-number>-<short-description>`

## 2. Implement the solution step-by-step

- Follow the task breakdown from your scratchpad plan.
- Make focused, incremental changes—one task at a time.

## 3. Commit after each step

- Use descriptive commit messages.
- Include the issue number in each commit message.

---

# TEST

- Use `pytest` to run all backend unit and integration tests.
- Write new tests for any added or changed functionality.
- Run the FastMCP verification script to ensure tools are correctly registered.
- Use FastAPI test client or `httpx` to simulate endpoint requests.
- Check LangGraph workflows return expected results and handle failures.
- If tests fail, debug and fix before proceeding.
- Ensure all tests pass before opening a PR.

---

# DEPLOY

## 1. Push your branch to GitHub

## 2. Open a Pull Request

- **Title:** Brief summary of the change.
- **Body:**
  - Link to the issue.
  - Describe what was fixed or added.
  - List test steps or key behaviors validated.

## 3. Request a review

- Tag relevant team members or issue assignees.
