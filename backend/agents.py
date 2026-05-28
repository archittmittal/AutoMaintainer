import os
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from litellm import completion
from typing import TypedDict, Annotated
import asyncio
from github import Github
import uuid

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
gh = Github(GITHUB_TOKEN) if GITHUB_TOKEN else None

class AgentState(TypedDict):
    repo_name: str
    architect_directive: str
    idea: str
    pm_decision: str
    code: str
    review: str
    issue_number: int
    pr_number: int
    log_messages: list

def run_llm(system_prompt: str, user_prompt: str):
    response = completion(
        model="groq/llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        api_key=GROQ_API_KEY
    )
    return response.choices[0].message.content

def architect_node(state: AgentState):
    repo = state["repo_name"]
    tree_content = ""
    readme_content = ""
    
    state["log_messages"].append({"type": "ui_update", "agentStatus": {"Architect": "active"}})
    
    if gh:
        try:
            gh_repo = gh.get_repo(repo)
            # Fetch root directory files to assess tech stack
            contents = gh_repo.get_contents("")
            tree_content = "\n".join([c.path for c in contents])
            
            try:
                readme = gh_repo.get_readme()
                readme_content = readme.decoded_content.decode('utf-8')
            except:
                readme_content = "No README found."
                
        except Exception as e:
            tree_content = "Unable to fetch repo tree."
            readme_content = "Inaccessible."

    system_prompt = "You are the Principal Architect. Analyze the provided repository root file structure and README context. Assess the current state of the project (is it working, what tech stack is it using) and give a strict 2-sentence directive on what the team should build or fix next."
    user_prompt = f"Repo: {repo}\n\nFiles:\n{tree_content}\n\nREADME:\n{readme_content[:1000]}\n\nGenerate the architect_directive."
    
    directive = run_llm(system_prompt, user_prompt)
    state["architect_directive"] = directive
    
    state["log_messages"].append({
        "type": "ui_update",
        "pipeline": {"id": "#NEW", "title": directive[:30] + "...", "status": "architecting", "agent": "Architect"}
    })
    state["log_messages"].append({"agent": "Architect", "msg": f"Directive: {directive}", "color": "text-rose-400"})
    state["log_messages"].append({"type": "ui_update", "agentStatus": {"Architect": "idle"}})
    return state

def brainstormer_node(state: AgentState):
    repo = state["repo_name"]
    directive = state.get("architect_directive", "")
    
    system_prompt = "You are the Visionary Agent. Your job is to brainstorm one single, highly innovative feature that fulfills the Architect's directive."
    user_prompt = f"Architect Directive:\n{directive}\n\nBrainstorm a new feature for {repo}. Keep it under 3 sentences."
    
    idea = run_llm(system_prompt, user_prompt)
    state["idea"] = idea
    
    state["log_messages"].append({"type": "ui_update", "agentStatus": {"Visionary": "active"}})
    state["log_messages"].append({
        "type": "ui_update",
        "pipeline": {"id": "#NEW", "title": idea[:30] + "...", "status": "brainstorming", "agent": "Visionary"}
    })
    state["log_messages"].append({"agent": "Visionary", "msg": f"Proposed Feature: {idea}", "color": "text-emerald-400"})
    
    # GitHub Action: Create Issue
    if gh:
        try:
            gh_repo = gh.get_repo(repo)
            issue = gh_repo.create_issue(
                title="[Feature Request] AI Brainstormed Idea",
                body=f"### Architect Directive\n{directive}\n\n### Proposed Feature\n{idea}"
            )
            state["issue_number"] = issue.number
            state["log_messages"].append({"agent": "System", "msg": f"Created GitHub Issue #{issue.number}", "color": "text-emerald-500"})
        except Exception as e:
            state["log_messages"].append({"agent": "System", "msg": f"Failed to create Issue: {str(e)}", "color": "text-red-500"})

    state["log_messages"].append({"type": "ui_update", "agentStatus": {"Visionary": "idle"}})
    return state

def pm_node(state: AgentState):
    idea = state["idea"]
    directive = state.get("architect_directive", "")
    repo = state["repo_name"]
    issue_number = state.get("issue_number")
    
    system_prompt = "You are the Product Manager. Review the proposed feature against the Architect's directive. Decide if we should build it ('APPROVED') or not ('REJECTED'). Start your response with APPROVED or REJECTED, then give a 1 sentence reason."
    
    decision = run_llm(system_prompt, f"Directive: {directive}\n\nReview this idea: {idea}")
    state["pm_decision"] = decision
    
    state["log_messages"].append({"type": "ui_update", "agentStatus": {"Reviewer": "active"}})
    state["log_messages"].append({
        "type": "ui_update",
        "pipeline": {"id": f"#{issue_number}" if issue_number else "#NEW", "title": idea[:30] + "...", "status": "reviewing", "agent": "Reviewer"}
    })
    
    is_approved = decision.startswith("APPROVED")
    msg_color = "text-amber-400" if is_approved else "text-red-400"
    state["log_messages"].append({"agent": "Reviewer", "msg": f"Decision: {decision}", "color": msg_color})
    
    # GitHub Action: Comment on Issue
    if gh and issue_number:
        try:
            gh_repo = gh.get_repo(repo)
            issue = gh_repo.get_issue(number=issue_number)
            issue.create_comment(f"**Reviewer Decision:** {decision}")
            if not is_approved:
                issue.edit(state="closed")
            state["log_messages"].append({"agent": "System", "msg": f"Commented on Issue #{issue_number}", "color": "text-zinc-500"})
        except Exception as e:
            state["log_messages"].append({"agent": "System", "msg": f"Failed to comment on Issue: {str(e)}", "color": "text-red-500"})
            
    state["log_messages"].append({"type": "ui_update", "agentStatus": {"Reviewer": "idle"}})
    return state

def should_implement(state: AgentState):
    return "implementer" if state.get("pm_decision", "").startswith("APPROVED") else END

def implementer_node(state: AgentState):
    idea = state["idea"]
    issue_number = state.get("issue_number")
    
    system_prompt = "You are the Implementer Agent. Write a tiny python script that implements the core logic of the idea. Keep it very short."
    code = run_llm(system_prompt, f"Write code for: {idea}")
    state["code"] = code
    
    state["log_messages"].append({"type": "ui_update", "agentStatus": {"Implementer": "active"}})
    state["log_messages"].append({
        "type": "ui_update",
        "pipeline": {"id": f"#{issue_number}" if issue_number else "#NEW", "title": idea[:30] + "...", "status": "implementing", "agent": "Implementer"}
    })
    state["log_messages"].append({"agent": "Implementer", "msg": f"Generated code implementation.", "color": "text-blue-400"})
    
    # GitHub Action: Open PR linking to issue
    if gh and issue_number:
        try:
            gh_repo = gh.get_repo(state["repo_name"])
            default_branch = gh_repo.default_branch
            sb = gh_repo.get_branch(default_branch)
            
            branch_name = f"feature/issue-{issue_number}-{uuid.uuid4().hex[:4]}"
            gh_repo.create_git_ref(ref=f"refs/heads/{branch_name}", sha=sb.commit.sha)
            
            code_to_commit = code
            if "```python" in code:
                code_to_commit = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code_to_commit = code.split("```")[1].split("```")[0].strip()
                
            gh_repo.create_file(
                path=f"ai_feature_issue_{issue_number}.py",
                message=f"Auto-generated feature for Issue #{issue_number}",
                content=code_to_commit,
                branch=branch_name
            )
            
            pr = gh_repo.create_pull(
                title=f"[AI] Implement Feature from #{issue_number}",
                body=f"This PR was automatically generated.\n\nResolves #{issue_number}",
                head=branch_name,
                base=default_branch
            )
            state["pr_number"] = pr.number
            state["log_messages"].append({"agent": "System", "msg": f"Created PR #{pr.number}: {pr.html_url}", "color": "text-emerald-500"})
        except Exception as e:
            state["log_messages"].append({"agent": "System", "msg": f"Failed to create PR: {str(e)}", "color": "text-red-500"})

    state["log_messages"].append({"type": "ui_update", "agentStatus": {"Implementer": "idle"}})
    return state

def maintainer_node(state: AgentState):
    code = state["code"]
    repo_name = state["repo_name"]
    pr_number = state.get("pr_number")
    
    system_prompt = "You are the Maintainer. Review the code. Say 'LGTM - Merging PR' if it looks okay, or point out a flaw."
    review = run_llm(system_prompt, f"Review this code:\n{code}")
    state["review"] = review
    
    state["log_messages"].append({"type": "ui_update", "agentStatus": {"Maintainer": "active"}})
    state["log_messages"].append({"agent": "Maintainer", "msg": f"Code Review: {review}", "color": "text-purple-400"})
    
    is_lgtm = "LGTM" in review or "Merging" in review
    
    # GitHub Action: Comment on PR and potentially merge
    if gh and pr_number:
        try:
            gh_repo = gh.get_repo(repo_name)
            pr = gh_repo.get_pull(pr_number)
            pr.create_issue_comment(f"**Maintainer Review:**\n{review}")
            
            if is_lgtm:
                pr.merge(commit_message=f"Auto-merged by AI Maintainer (PR #{pr_number})")
                state["log_messages"].append({
                    "type": "ui_update",
                    "activity": {"title": f"Merged PR #{pr_number}", "time": "Just now", "type": "merge"}
                })
                state["log_messages"].append({"agent": "System", "msg": f"Successfully merged PR #{pr_number}!", "color": "text-emerald-500"})
        except Exception as e:
            state["log_messages"].append({"agent": "System", "msg": f"Failed to review/merge PR: {str(e)}", "color": "text-red-500"})

    state["log_messages"].append({"type": "ui_update", "agentStatus": {"Maintainer": "idle"}})
    return state

# Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("architect", architect_node)
workflow.add_node("brainstormer", brainstormer_node)
workflow.add_node("pm", pm_node)
workflow.add_node("implementer", implementer_node)
workflow.add_node("maintainer", maintainer_node)

workflow.set_entry_point("architect")
workflow.add_edge("architect", "brainstormer")
workflow.add_edge("brainstormer", "pm")
workflow.add_conditional_edges("pm", should_implement)
workflow.add_edge("implementer", "maintainer")
workflow.add_edge("maintainer", END)

app = workflow.compile()

async def run_agent_loop(repo_name: str, ws_manager):
    if "github.com/" in repo_name:
        repo_name = repo_name.split("github.com/")[-1].strip("/")
    
    initial_state = {
        "repo_name": repo_name, 
        "architect_directive": "",
        "idea": "", 
        "pm_decision": "", 
        "code": "", 
        "review": "", 
        "issue_number": 0,
        "pr_number": 0,
        "log_messages": []
    }
    
    await ws_manager.broadcast({"agent": "System", "msg": f"Starting loop for repo: {repo_name}...", "color": "text-zinc-500"})
    
    last_idx = 0
    for s in app.stream(initial_state):
        node_name = list(s.keys())[0]
        state = s[node_name]
        
        new_msgs = state["log_messages"][last_idx:]
        for msg in new_msgs:
            await ws_manager.broadcast(msg)
            await asyncio.sleep(0.5)
            
        last_idx = len(state["log_messages"])

    await ws_manager.broadcast({"agent": "System", "msg": "Agent loop complete.", "color": "text-zinc-500"})
