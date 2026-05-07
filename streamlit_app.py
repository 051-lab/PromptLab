"""Streamlit frontend for PromptLab - LLM Prompt Engineering Platform."""

import asyncio
from datetime import datetime
from typing import Optional

import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="PromptLab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API base URL
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")


def mask_api_key(key: Optional[str]) -> str:
    """Mask API key for display, showing only first and last 4 characters."""
    if not key:
        return "Not configured"
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


def fetch_json(url: str, params: Optional[dict] = None) -> dict | list:
    """Fetch JSON from API endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}{url}", params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return []


def post_json(url: str, data: dict) -> dict | None:
    """Post JSON to API endpoint."""
    try:
        response = requests.post(f"{API_BASE_URL}{url}", json=data, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        st.warning("Request timed out. Large models may take longer to respond.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
        return None


def delete_request(url: str) -> bool:
    """Delete resource via API."""
    try:
        response = requests.delete(f"{API_BASE_URL}{url}", timeout=30)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Delete request failed: {e}")
        return False


# Available models for comparison
AVAILABLE_MODELS = {
    "gpt-4o": "GPT-4o (OpenAI)",
    "gpt-4-turbo": "GPT-4 Turbo (OpenAI)",
    "gpt-3.5-turbo": "GPT-3.5 Turbo (OpenAI)",
    "claude-3-opus-20240229": "Claude 3 Opus (Anthropic)",
    "claude-3-sonnet-20240229": "Claude 3 Sonnet (Anthropic)",
    "claude-3-haiku-20240307": "Claude 3 Haiku (Anthropic)",
    "gemini/gemini-pro": "Gemini Pro (Google)",
    "gemini/gemini-1.5-pro": "Gemini 1.5 Pro (Google)",
    "qwen/qwen-max": "Qwen Max (Alibaba)",
    "qwen/qwen-plus": "Qwen Plus (Alibaba)",
}


def render_home():
    """Render the home page with project description."""
    st.title("🧪 PromptLab")
    st.markdown("### A Production-Ready LLM Prompt Engineering Platform")

    st.markdown("""
    **PromptLab** is a comprehensive platform for designing, testing, and comparing 
    prompts across multiple Large Language Models (LLMs).
    
    #### Features
    
    - **📝 Prompt Library**: Create, edit, and organize your prompt templates
    - **⚖️ Model Comparison**: Run side-by-side comparisons across different LLM providers
    - **📊 Experiment Analytics**: Track latency, token usage, and output quality
    - **🔧 Production Ready**: Built with FastAPI, SQLAlchemy, and Streamlit
    
    #### Supported Models
    
    | Provider | Models |
    |----------|--------|
    | OpenAI | GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo |
    | Anthropic | Claude 3 Opus, Sonnet, Haiku |
    | Google | Gemini Pro, Gemini 1.5 Pro |
    | Alibaba | Qwen Max, Qwen Plus |
    
    #### Getting Started
    
    1. Navigate to **Prompt Library** to create your first prompt
    2. Use **Compare Models** to test prompts across different LLMs
    3. Review results in **Experiment History** with detailed analytics
    """)

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Supported Providers", "4")
    with col2:
        st.metric("Available Models", len(AVAILABLE_MODELS))
    with col3:
        st.metric("API Status", "🟢 Online")
    with col4:
        st.metric("Version", "0.3.0")


def render_prompt_library():
    """Render the prompt library management interface."""
    st.title("📝 Prompt Library")
    st.markdown("Manage your prompt templates")

    # Sidebar for actions
    with st.sidebar:
        st.header("Actions")
        action = st.radio("Choose action", ["View All", "Create New", "Search"])

    # Fetch prompts
    prompts = fetch_json("/api/v1/prompts")

    if action == "Create New":
        st.subheader("Create New Prompt")
        with st.form("create_prompt_form"):
            name = st.text_input("Name *", placeholder="e.g., Customer Support Response")
            description = st.text_area("Description", placeholder="Optional description...")
            template = st.text_area("Template *", height=200, placeholder="Enter your prompt template...")
            model_name = st.selectbox("Default Model", list(AVAILABLE_MODELS.keys()))
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
            max_tokens = st.number_input("Max Tokens", 1, 100000, 2048)

            submitted = st.form_submit_button("Create Prompt", use_container_width=True)

            if submitted:
                if not name or not template:
                    st.error("Name and Template are required fields")
                else:
                    payload = {
                        "name": name,
                        "description": description if description else None,
                        "template": template,
                        "model_name": model_name,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                    result = post_json("/api/v1/prompts", payload)
                    if result:
                        st.success(f"Prompt '{name}' created successfully!")
                        st.rerun()

    elif action == "Search":
        st.subheader("Search Prompts")
        search_term = st.text_input("Search by name or description")
        if search_term and prompts:
            filtered = [
                p for p in prompts
                if search_term.lower() in p.get("name", "").lower()
                or search_term.lower() in p.get("description", "").lower()
            ]
            prompts = filtered
            st.info(f"Found {len(prompts)} matching prompts")

    # Display prompts
    if not prompts:
        st.info("No prompts found. Create your first prompt!")
    else:
        st.subheader(f"All Prompts ({len(prompts)})")
        for prompt in prompts:
            with st.expander(f"📄 {prompt['name']}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    if prompt.get("description"):
                        st.markdown(f"*{prompt['description']}*")
                    st.code(prompt["template"], language="text")
                with col2:
                    st.markdown("**Settings:**")
                    st.write(f"Model: `{prompt['model_name']}`")
                    st.write(f"Temp: {prompt['temperature']}")
                    st.write(f"Max Tokens: {prompt['max_tokens']}")
                    st.write(f"Created: {prompt['created_at'][:10]}")

                    # Edit/Delete buttons
                    if st.button("Edit", key=f"edit_{prompt['id']}"):
                        st.session_state["editing_prompt"] = prompt
                        st.rerun()
                    if st.button("Delete", key=f"delete_{prompt['id']}", type="secondary"):
                        if delete_request(f"/api/v1/prompts/{prompt['id']}"):
                            st.success("Prompt deleted!")
                            st.rerun()

    # Handle editing
    if "editing_prompt" in st.session_state:
        prompt = st.session_state["editing_prompt"]
        st.subheader(f"Edit Prompt: {prompt['name']}")
        with st.form("edit_prompt_form"):
            name = st.text_input("Name", value=prompt["name"])
            description = st.text_area("Description", value=prompt.get("description") or "")
            template = st.text_area("Template", value=prompt["template"], height=200)
            model_name = st.selectbox(
                "Default Model",
                list(AVAILABLE_MODELS.keys()),
                index=list(AVAILABLE_MODELS.keys()).index(prompt["model_name"])
                if prompt["model_name"] in AVAILABLE_MODELS else 0
            )
            temperature = st.slider("Temperature", 0.0, 2.0, prompt["temperature"], 0.1)
            max_tokens = st.number_input("Max Tokens", 1, 100000, prompt["max_tokens"])

            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("Save Changes", use_container_width=True)
            with col2:
                cancelled = st.form_submit_button("Cancel", use_container_width=True)

            if submitted:
                payload = {
                    "name": name,
                    "description": description if description else None,
                    "template": template,
                    "model_name": model_name,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                result = requests.put(
                    f"{API_BASE_URL}/api/v1/prompts/{prompt['id']}",
                    json=payload,
                    timeout=30
                )
                if result.status_code == 200:
                    st.success("Prompt updated successfully!")
                    del st.session_state["editing_prompt"]
                    st.rerun()
                else:
                    st.error(f"Update failed: {result.text}")

            if cancelled:
                del st.session_state["editing_prompt"]
                st.rerun()


def render_compare():
    """Render the model comparison interface."""
    st.title("⚖️ Compare Models")
    st.markdown("Run side-by-side comparisons across different LLM providers")

    # Input method selection
    input_method = st.radio("Input Method", ["Select from Library", "Enter Custom Text"])

    prompt_text = None
    selected_prompt = None

    if input_method == "Select from Library":
        prompts = fetch_json("/api/v1/prompts")
        if prompts:
            prompt_options = {p["name"]: p for p in prompts}
            selected_name = st.selectbox("Select a prompt", list(prompt_options.keys()))
            if selected_name:
                selected_prompt = prompt_options[selected_name]
                st.info(f"**Template:** {selected_prompt['template'][:200]}...")
                prompt_text = selected_prompt["template"]
        else:
            st.warning("No prompts available. Create a prompt first or use custom text.")
            input_method = "Enter Custom Text"
            prompt_text = st.text_area("Enter your prompt", height=150)
    else:
        prompt_text = st.text_area("Enter your prompt", height=150, placeholder="Type your prompt here...")

    # Model selection
    st.subheader("Select Models to Compare")
    st.markdown("Choose 2 or more models for comparison")

    cols = st.columns(3)
    selected_models = []
    for i, (model_id, model_name) in enumerate(AVAILABLE_MODELS.items()):
        with cols[i % 3]:
            if st.checkbox(model_name, value=False, key=model_id):
                selected_models.append(model_id)

    if len(selected_models) < 2:
        st.warning(f"Please select at least 2 models (currently selected: {len(selected_models)})")

    # Settings
    with st.expander("Advanced Settings"):
        temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        max_tokens = st.number_input("Max Tokens", 1, 100000, 2048)
        timeout = st.number_input("Timeout (seconds)", 30, 300, 120)

    # Run comparison
    if st.button("🚀 Run Comparison", type="primary", disabled=len(selected_models) < 2 or not prompt_text):
        if not prompt_text:
            st.error("Please enter or select a prompt")
        elif len(selected_models) < 2:
            st.error("Please select at least 2 models")
        else:
            with st.spinner("Running comparison across models..."):
                # Prepare request
                results_payload = [
                    {"model_name": model, "output": "", "latency_ms": 0, "tokens_used": 0, "success": True}
                    for model in selected_models
                ]
                payload = {
                    "prompt_text": prompt_text,
                    "prompt_id": selected_prompt["id"] if selected_prompt else None,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "results": results_payload,
                }

                # Call API
                experiment = post_json("/api/v1/compare/compare", payload)

                if experiment:
                    st.success("Comparison completed!")
                    display_comparison_results(experiment)


def display_comparison_results(experiment: dict):
    """Display comparison results in a table with metrics."""
    st.subheader("Results")

    results = experiment.get("results", [])
    if not results:
        st.warning("No results returned")
        return

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    successful = sum(1 for r in results if r.get("success", False))
    avg_latency = sum(r.get("latency_ms", 0) for r in results if r.get("success")) / max(successful, 1)
    total_tokens = sum(r.get("tokens_used", 0) for r in results if r.get("success"))

    with col1:
        st.metric("Successful", f"{successful}/{len(results)}")
    with col2:
        st.metric("Avg Latency", f"{avg_latency:.0f}ms")
    with col3:
        st.metric("Total Tokens", total_tokens)

    # Results table
    st.markdown("### Detailed Results")

    for result in results:
        model_display = AVAILABLE_MODELS.get(result["model_name"], result["model_name"])
        status_icon = "✅" if result.get("success") else "❌"

        with st.expander(f"{status_icon} {model_display}"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("**Output:**")
                if result.get("success"):
                    st.write(result.get("output", "No output"))
                else:
                    st.error(f"Error: {result.get('error_message', 'Unknown error')}")
            with col2:
                st.markdown("**Metrics:**")
                st.write(f"Latency: {result.get('latency_ms', 0):.2f}ms")
                st.write(f"Tokens: {result.get('tokens_used', 0)}")
                st.write(f"Status: {'Success' if result.get('success') else 'Failed'}")


def render_experiment_history():
    """Render experiment history with analytics charts."""
    st.title("📊 Experiment History")
    st.markdown("View past comparison runs and analytics")

    # Fetch experiments
    experiments = fetch_json("/api/v1/compare", params={"limit": 50})

    if not experiments:
        st.info("No experiments found. Run a comparison to see results here.")
        return

    # Experiment list
    st.subheader("Recent Experiments")

    selected_exp = st.selectbox(
        "Select an experiment to view details",
        options=[f"#{exp['id']} - {exp['prompt_text'][:50]}..." for exp in experiments],
        format_func=lambda x: x
    )

    if selected_exp:
        exp_id = int(selected_exp.split(" - ")[0].replace("#", ""))
        experiment = fetch_json(f"/api/v1/compare/{exp_id}")

        if experiment:
            # Display experiment details
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Models Tested", len(experiment.get("results", [])))
            with col2:
                st.metric("Temperature", experiment.get("temperature", "N/A"))
            with col3:
                st.metric("Max Tokens", experiment.get("max_tokens", "N/A"))

            st.markdown("**Prompt:**")
            st.code(experiment.get("prompt_text", ""), language="text")

            # Charts
            results = experiment.get("results", [])
            if results:
                st.subheader("Analytics")

                # Latency bar chart
                latency_data = {
                    "Model": [AVAILABLE_MODELS.get(r["model_name"], r["model_name"]) for r in results if r.get("success")],
                    "Latency (ms)": [r["latency_ms"] for r in results if r.get("success")]
                }
                if latency_data["Model"]:
                    fig_latency = px.bar(
                        latency_data,
                        x="Model",
                        y="Latency (ms)",
                        title="Response Latency by Model",
                        color="Latency (ms)",
                        color_continuous_scale="Viridis"
                    )
                    fig_latency.update_layout(xaxis_tickangle=-45, height=400)
                    st.plotly_chart(fig_latency, use_container_width=True)

                # Token usage comparison
                token_data = {
                    "Model": [AVAILABLE_MODELS.get(r["model_name"], r["model_name"]) for r in results if r.get("success")],
                    "Tokens Used": [r["tokens_used"] for r in results if r.get("success")]
                }
                if token_data["Model"]:
                    fig_tokens = px.bar(
                        token_data,
                        x="Model",
                        y="Tokens Used",
                        title="Token Usage by Model",
                        color="Tokens Used",
                        color_continuous_scale="Plasma"
                    )
                    fig_tokens.update_layout(xaxis_tickangle=-45, height=400)
                    st.plotly_chart(fig_tokens, use_container_width=True)

                # Side-by-side comparison table
                st.subheader("Comparison Table")
                table_data = []
                for r in results:
                    table_data.append({
                        "Model": AVAILABLE_MODELS.get(r["model_name"], r["model_name"]),
                        "Status": "✅ Success" if r.get("success") else "❌ Failed",
                        "Latency (ms)": f"{r.get('latency_ms', 0):.2f}" if r.get("success") else "N/A",
                        "Tokens": r.get("tokens_used", 0) if r.get("success") else "N/A",
                        "Output Preview": (r.get("output", "")[:100] + "...") if r.get("success") else r.get("error_message", "Error")
                    })
                st.table(table_data)

    # Delete option
    st.divider()
    if st.button("🗑️ Clear All Experiments", type="secondary"):
        if st.confirm("Are you sure you want to delete all experiments?"):
            for exp in experiments:
                delete_request(f"/api/v1/compare/{exp['id']}")
            st.success("All experiments deleted!")
            st.rerun()


def render_settings():
    """Render settings page showing API key configuration."""
    st.title("⚙️ Settings")
    st.markdown("View configured API keys and system settings")

    # API Keys section
    st.subheader("🔑 API Keys Configuration")
    st.markdown("""
    API keys are loaded from environment variables. Configure these before starting the application:
    
    - `OPENAI_API_KEY` - For OpenAI models (GPT-4, GPT-3.5)
    - `ANTHROPIC_API_KEY` - For Anthropic models (Claude)
    - `GOOGLE_API_KEY` - For Google models (Gemini)
    - `QWEN_API_KEY` - For Alibaba models (Qwen)
    """)

    # Try to get settings from API health endpoint
    health_info = fetch_json("/api/v1/health")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**OpenAI API Key**")
        openai_key = st.secrets.get("OPENAI_API_KEY")
        st.code(mask_api_key(openai_key), language="text")

    with col2:
        st.markdown("**Anthropic API Key**")
        anthropic_key = st.secrets.get("ANTHROPIC_API_KEY")
        st.code(mask_api_key(anthropic_key), language="text")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Google API Key**")
        google_key = st.secrets.get("GOOGLE_API_KEY")
        st.code(mask_api_key(google_key), language="text")

    with col4:
        st.markdown("**Qwen API Key**")
        qwen_key = st.secrets.get("QWEN_API_KEY")
        st.code(mask_api_key(qwen_key), language="text")

    # System info
    st.divider()
    st.subheader("ℹ️ System Information")

    if health_info:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Status", health_info.get("status", "Unknown"))
        with col2:
            st.metric("Version", health_info.get("version", "Unknown"))
        with col3:
            st.metric("Environment", health_info.get("environment", "Unknown"))

    st.markdown("""
    ### Setup Instructions
    
    1. Create a `.streamlit/secrets.toml` file in your project root
    2. Add your API keys:
    ```toml
    OPENAI_API_KEY = "your-openai-key"
    ANTHROPIC_API_KEY = "your-anthropic-key"
    GOOGLE_API_KEY = "your-google-key"
    QWEN_API_KEY = "your-qwen-key"
    API_BASE_URL = "http://localhost:8000"
    ```
    """)


def main():
    """Main Streamlit application."""
    # Sidebar navigation
    with st.sidebar:
        st.image("https://img.shields.io/badge/PromptLab-v0.3.0-blue", use_container_width=True)
        st.markdown("---")

        navigation = st.radio(
            "Navigation",
            ["Home", "Prompt Library", "Compare Models", "Experiment History", "Settings"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### Quick Links")
        st.markdown("[📚 API Docs](http://localhost:8000/docs)")
        st.markdown("[🔴 Redoc](http://localhost:8000/redoc)")

    # Route to appropriate page
    if navigation == "Home":
        render_home()
    elif navigation == "Prompt Library":
        render_prompt_library()
    elif navigation == "Compare Models":
        render_compare()
    elif navigation == "Experiment History":
        render_experiment_history()
    elif navigation == "Settings":
        render_settings()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>Built with FastAPI + Streamlit | PromptLab © 2024</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
