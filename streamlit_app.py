import os
import ast

import streamlit as st
import pandas as pd
import plotly.express as px

import numpy as np
import textwrap
from sklearn.decomposition import PCA
from openai import AzureOpenAI
from dotenv import load_dotenv


# ---------- 0. Azure OpenAI client + LLM helpers ----------

# Load environment variables from the .env file
load_dotenv()


@st.cache_resource
def get_azure_client():
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")

    if not all([api_version, azure_endpoint, api_key]):
        raise RuntimeError(
            "Missing one or more Azure OpenAI environment variables. "
            "Check AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY."
        )

    return AzureOpenAI(
        api_version=api_version,
        azure_endpoint=azure_endpoint,
        api_key=api_key,
    )


client = get_azure_client()

N_PER_CLUSTER = 10  # how many patents per cluster to show the LLM


def _sample_cluster(
    df_all: pd.DataFrame,
    cluster_id,
    n_per_cluster: int = N_PER_CLUSTER,
) -> pd.DataFrame:
    """Sample up to n_per_cluster patents for a given numeric cluster id."""
    if "cluster" not in df_all.columns or "embed_text" not in df_all.columns:
        return pd.DataFrame()

    subset = df_all.loc[
        (df_all["cluster"] == cluster_id) & df_all["embed_text"].notna(),
    ]

    if subset.empty:
        return subset

    n = min(n_per_cluster, len(subset))
    return subset.sample(n=n, random_state=42)


# ---------- 1. Load data ----------


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("nestle_patents.csv")

    # Make publication_date datetime if present
    if "publication_date" in df.columns:
        df["publication_date"] = pd.to_datetime(
            df["publication_date"],
            errors="coerce",
        )
    return df


@st.cache_data(show_spinner=False)
def get_cluster_summary(cluster_id: int) -> str:
    """
    Call the LLM once per cluster to get a textual summary.
    Cached by cluster_id so you don't pay twice.
    """
    df_all = load_data()  # use the same loader as the app
    sample_df = _sample_cluster(df_all, cluster_id)

    if sample_df.empty:
        return "No patents with embed_text found for this cluster."

    # Build compact text block
    docs = []
    for i, (_, row) in enumerate(sample_df.iterrows(), start=1):
        text = str(row["embed_text"])
        text = textwrap.shorten(text, width=2000, placeholder=" ...")
        docs.append(f"PATENT {i}:\n{text}")

    docs_block = "\n\n-----\n\n".join(docs)

    prompt = f"""
You are analysing Nestlé patent clusters as an expert.

Below are several representative patents from CLUSTER {cluster_id}.
Each PATENT text may include title, abstract, claim snippet and IPC codes.

Your task:
1. Identify the main technical/theme area for this cluster in 1 short label.
2. Give 3–5 bullet points describing what kinds of innovations appear here
   (e.g. ingredients, target population, applications, technologies).
3. Mention the main product categories (e.g. infant formula, coffee, packaging, pet food, supplements, etc.).

Respond in this structure (no extra explanation before or after):

CLUSTER_LABEL: <short label>

THEMES:
- <bullet 1>
- <bullet 2>
- <bullet 3>
- <bullet 4>
- <bullet 5>

MAIN_PRODUCT_CATEGORIES:
- <category 1>
- <category 2>
- <category 3>

Here are the patents:

{docs_block}
""".strip()

    resp = client.chat.completions.create(
        model="gpt-5.1",  # your Azure deployment name
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=300,
        temperature=0.1,
    )

    return resp.choices[0].message.content.strip()


df = load_data()

# ---------- 2. Page title ----------

st.title("Nestlé Patent Innovation Dashboard")

st.write(
    "This dashboard summarises Nestlé's patent portfolio: "
    "publication trends, legal status mix, and cluster-level insights."
)

# ---------- 3. Sidebar filters ----------

st.sidebar.header("Filters")

# --- Year filter (only if pub_year exists) ---
if "pub_year" in df.columns:
    years = sorted(df["pub_year"].dropna().unique())
    if len(years) > 0:
        min_year, max_year = int(min(years)), int(max(years))
        year_range = st.sidebar.slider(
            "Publication year range",
            min_year,
            max_year,
            (min_year, max_year),
        )
    else:
        year_range = None
else:
    st.sidebar.warning("Column 'pub_year' not found; year filter disabled.")
    year_range = None

# --- Legal status filter (only if legal_status_clean exists) ---
if "legal_status_clean" in df.columns:
    status_options = sorted(df["legal_status_clean"].dropna().unique())
    selected_status = st.sidebar.multiselect(
        "Legal status",
        options=status_options,
        default=status_options,  # all by default
    )
else:
    st.sidebar.warning("Column 'legal_status_clean' not found; legal status filter disabled.")
    selected_status = None

# --- Optional: cluster filter if you have cluster_text ---
if "cluster_text" in df.columns:
    cluster_options = sorted(df["cluster_text"].dropna().unique())
    selected_clusters = st.sidebar.multiselect(
        "Cluster",
        options=cluster_options,
        default=cluster_options,
    )
else:
    selected_clusters = None

# ---------- 3b. Apply filters ----------

mask = pd.Series(True, index=df.index)

if year_range is not None and "pub_year" in df.columns:
    mask &= (df["pub_year"] >= year_range[0]) & (df["pub_year"] <= year_range[1])

if selected_status is not None and "legal_status_clean" in df.columns:
    mask &= df["legal_status_clean"].isin(selected_status)

if selected_clusters is not None and "cluster_text" in df.columns:
    mask &= df["cluster_text"].isin(selected_clusters)

df_filtered = df.loc[mask].copy()

# ---------- 4+. Main content ----------

if df_filtered.empty:
    st.warning("No data matches the current filters.")
else:
    # ---------- 4. Summary metrics ----------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Number of patents", len(df_filtered))

    with col2:
        if "pub_year" in df_filtered.columns:
            st.metric("Distinct publication years", df_filtered["pub_year"].nunique())
        else:
            st.metric("Distinct publication years", "N/A")

    with col3:
        if "cluster_text" in df_filtered.columns:
            st.metric("Distinct clusters", df_filtered["cluster_text"].nunique())
        elif "cluster" in df_filtered.columns:
            st.metric("Distinct clusters", df_filtered["cluster"].nunique())
        else:
            st.metric("Distinct clusters", "N/A")

    # ---------- 5. Plots ----------

    # 5.1 Publications over time
    if "pub_year" in df_filtered.columns:
        st.subheader("Publications over time")

        pub_counts = (
            df_filtered.groupby("pub_year")
            .size()
            .reset_index(name="count")
            .sort_values("pub_year")
        )

        if not pub_counts.empty:
            fig_years = px.bar(
                pub_counts,
                x="pub_year",
                y="count",
                labels={"pub_year": "Publication year", "count": "Number of patents"},
                title="Nestlé patent publications per year",
            )
            st.plotly_chart(fig_years, width="stretch")
        else:
            st.info("No publication year data available for current filters.")
    else:
        st.info("Column 'pub_year' not available – skipping publications over time plot.")

    # 5.2 Legal status distribution
    if "legal_status_clean" in df_filtered.columns:
        st.subheader("Legal status distribution")

        status_counts = df_filtered["legal_status_clean"].value_counts().reset_index()
        status_counts.columns = ["legal_status_clean", "count"]

        if not status_counts.empty:
            fig_status = px.pie(
                status_counts,
                names="legal_status_clean",
                values="count",
                title="Legal status share (filtered subset)",
            )
            st.plotly_chart(fig_status, width="stretch")
        else:
            st.info("No legal status data available for current filters.")
    else:
        st.info("Column 'legal_status_clean' not available – skipping legal status plot.")

    # 5.3 Patents per cluster (using cluster_text if present)
    if "cluster_text" in df_filtered.columns:
        st.subheader("Patents per cluster")

        cluster_counts = df_filtered["cluster_text"].value_counts().reset_index()
        cluster_counts.columns = ["cluster_text", "count"]

        if not cluster_counts.empty:
            fig_clusters = px.bar(
                cluster_counts,
                x="cluster_text",
                y="count",
                labels={"cluster_text": "Cluster", "count": "Number of patents"},
                title="Patents by cluster (filtered subset)",
            )
            fig_clusters.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_clusters, width="stretch")
        else:
            st.info("No cluster information available for current filters.")

    # ---------- 6. Embedding space (PCA projection) ----------

    if "embeddings" in df_filtered.columns:
        st.subheader("Embedding space (PCA projection)")

        emb_series = df_filtered["embeddings"].dropna()

        def parse_embedding(x):
            if isinstance(x, (list, np.ndarray)):
                return np.asarray(x, dtype=float)
            if isinstance(x, str):
                try:
                    return np.asarray(ast.literal_eval(x), dtype=float)
                except Exception:
                    return None
            return None

        emb_arrays = emb_series.apply(parse_embedding).dropna()

        if len(emb_arrays) >= 2:
            emb_df = df_filtered.loc[emb_arrays.index].copy()
            E = np.vstack(emb_arrays.to_list())

            pca = PCA(n_components=2, random_state=0)
            E_2d = pca.fit_transform(E)

            emb_df["PC1"] = E_2d[:, 0]
            emb_df["PC2"] = E_2d[:, 1]

            color_col = None
            color_label = None
            if "cluster_text" in emb_df.columns:
                emb_df["cluster_text"] = emb_df["cluster_text"].fillna("Unlabelled")
                color_col = "cluster_text"
                color_label = "Cluster"
            elif "cluster" in emb_df.columns:
                emb_df["cluster"] = emb_df["cluster"].fillna(-1)
                color_col = "cluster"
                color_label = "Cluster ID"

            x_min, x_max = emb_df["PC1"].min(), emb_df["PC1"].max()
            y_min, y_max = emb_df["PC2"].min(), emb_df["PC2"].max()
            x_pad = 0.05 * (x_max - x_min)
            y_pad = 0.05 * (y_max - y_min)
            range_x = [x_min - x_pad, x_max + x_pad]
            range_y = [y_min - y_pad, y_max + y_pad]

            view_mode = st.select_slider(
                "PCA view",
                options=["Unlabelled", "Coloured by cluster"],
                value="Unlabelled",
            )

            if view_mode == "Unlabelled" or color_col is None:
                fig = px.scatter(
                    emb_df,
                    x="PC1",
                    y="PC2",
                    title="Patent embeddings (PCA) – no cluster colours",
                    labels={"PC1": "PC1", "PC2": "PC2"},
                    range_x=range_x,
                    range_y=range_y,
                )
            else:
                fig = px.scatter(
                    emb_df,
                    x="PC1",
                    y="PC2",
                    color=color_col,
                    title="Patent embeddings (PCA) – coloured by cluster",
                    labels={"PC1": "PC1", "PC2": "PC2", color_col: color_label},
                    range_x=range_x,
                    range_y=range_y,
                )

            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Not enough valid embeddings to compute PCA projection for the current filters.")
    else:
        st.info("Column 'embeddings' not available – skipping embedding PCA plots.")

    # ---------- 7. Cluster inspector (LLM summary) ----------

    if ("cluster" in df.columns) and ("embed_text" in df.columns):
        st.subheader("Cluster inspector (LLM summary)")

        if "cluster_text" in df.columns:
            cluster_map = (
                df[["cluster", "cluster_text"]]
                .dropna()
                .drop_duplicates()
                .set_index("cluster")["cluster_text"]
                .to_dict()
            )
        else:
            cluster_map = {}

        cluster_ids = sorted(df["cluster"].dropna().unique())
        if len(cluster_ids) > 0:
            selected_cluster_id = st.selectbox(
                "Select cluster to summarise",
                options=cluster_ids,
                format_func=lambda cid: f"{cid} – {cluster_map.get(cid, f'Cluster {cid}')}",
            )

            if st.button("Generate / refresh LLM summary"):
                with st.spinner("Calling LLM to summarise this cluster..."):
                    summary_text = get_cluster_summary(int(selected_cluster_id))

                st.markdown("**LLM cluster summary:**")
                st.markdown(f"```text\n{summary_text}\n```")

                if "cluster" in df_filtered.columns:
                    st.markdown("**Sample patents from this cluster (filtered view):**")
                    cols_to_show = [
                        c
                        for c in ["pub_year", "legal_status_clean", "cluster_text", "embed_text"]
                        if c in df_filtered.columns
                    ]
                    if cols_to_show:
                        st.dataframe(
                            df_filtered[df_filtered["cluster"] == selected_cluster_id][
                                cols_to_show
                            ].head(10)
                        )
        else:
            st.info("No clusters found in data to inspect.")
    else:
        st.info("Columns 'cluster' and/or 'embed_text' not available – skipping cluster inspector.")

    # ---------- 8. Novelty distribution by cluster ----------

    if {"novelty_score", "cluster_text"}.issubset(df_filtered.columns):
        st.subheader("Novelty distribution by cluster")

        for cluster_text_val, group in df_filtered.groupby("cluster_text"):
            st.markdown(f"{cluster_text_val}")
            if group["novelty_score"].notna().any():
                fig_nov = px.histogram(
                    group,
                    x="novelty_score",
                    nbins=20,
                    title=f"Novelty scores – {cluster_text_val}",
                    marginal="box",
                )
                st.plotly_chart(fig_nov, width="stretch")
            else:
                st.info(f"No novelty scores available for cluster '{cluster_text_val}'.")
    else:
        st.info("Columns 'novelty_score' and/or 'cluster_text' not available – skipping novelty section.")

    # ---------- 9. Average patent age by legal status ----------

    if "publication_date" in df_filtered.columns and "legal_status_clean" in df_filtered.columns:
        st.subheader("Average patent age by legal status")

        today = pd.Timestamp.today()
        valid_mask = df_filtered["publication_date"].notna()
        df_filtered.loc[valid_mask, "age_years"] = (
            (today - df_filtered.loc[valid_mask, "publication_date"]).dt.days / 365.25
        )

        age_df = (
            df_filtered.dropna(subset=["age_years", "legal_status_clean"])
            .groupby("legal_status_clean")["age_years"]
            .mean()
            .reset_index()
            .sort_values("age_years")
        )

        if not age_df.empty:
            fig_age = px.bar(
                age_df,
                x="age_years",
                y="legal_status_clean",
                orientation="h",
                labels={
                    "age_years": "Average age (years)",
                    "legal_status_clean": "Legal status",
                },
                title="Average patent age by legal status (filtered subset)",
            )
            st.plotly_chart(fig_age, width="stretch")
            st.dataframe(age_df.style.format({"age_years": "{:.2f}"}))
        else:
            st.info("No age data available for current filters.")
    else:
        st.info(
            "Columns 'publication_date' and/or 'legal_status_clean' not available – "
            "skipping age analysis."
        )

    # ---------- 10. Patents per cluster by legal status (heatmap) ----------

    if "legal_status_clean" in df_filtered.columns and (
        "cluster_text" in df_filtered.columns or "cluster" in df_filtered.columns
    ):
        st.subheader("Patents per cluster by legal status")

        if "cluster_text" in df_filtered.columns:
            cluster_col = "cluster_text"
            cluster_label = "Cluster"
        else:
            cluster_col = "cluster"
            cluster_label = "Cluster ID"

        status_by_cluster = (
            df_filtered.groupby([cluster_col, "legal_status_clean"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )

        if not status_by_cluster.empty:
            fig_heat = px.imshow(
                status_by_cluster,
                aspect="auto",
                labels={
                    "x": "Legal status",
                    "y": cluster_label,
                    "color": "Number of patents",
                },
                title="Patents per cluster by legal status (filtered subset)",
            )
            st.plotly_chart(fig_heat, width="stretch")
            st.dataframe(status_by_cluster)
        else:
            st.info(
                "No data available to compute cluster × legal status heatmap "
                "for current filters."
            )
    else:
        st.info(
            "Columns for clusters and/or legal status not available – "
            "skipping 'patents per cluster by legal status' heatmap."
        )
