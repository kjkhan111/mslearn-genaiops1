"""
Complete Cloud Evaluation Script for Trail Guide Agent

Runs the full evaluation pipeline:
1. Uploads evaluation dataset to Microsoft Foundry
2. Creates evaluation definition with quality evaluators
3. Runs evaluation and polls for completion
4. Retrieves and displays results

Evaluates:
- Intent Resolution
- Relevance
- Groundedness
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

from openai.types.eval_create_params import DataSourceConfigCustom
from openai.types.evals.create_eval_jsonl_run_data_source_param import (
    CreateEvalJSONLRunDataSourceParam,
    SourceFileID,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

# Reads variables from the .env file in your project root
endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
model_deployment_name = os.environ.get("MODEL_NAME", "gpt-5.1")

dataset_name = "trail-guide-evaluation-dataset"
dataset_version = "1"

# The script writes a plain-text summary here when it finishes.
# This file can be consumed by GitHub Actions.
RESULTS_FILE = Path("evaluation_results.txt")


if not endpoint:
    print("ERROR: AZURE_AI_PROJECT_ENDPOINT is not set.")
    print("       Add it to your .env file and try again.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Azure clients
# ---------------------------------------------------------------------------

# AIProjectClient connects to your Azure AI Foundry project
project_client = AIProjectClient(
    endpoint=endpoint,
    credential=DefaultAzureCredential(),
)

# OpenAI-compatible client exposes the Evals API
client = project_client.get_openai_client()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    """Print a clearly visible section header."""
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")


# ---------------------------------------------------------------------------
# Step 1 – Upload the evaluation dataset
# ---------------------------------------------------------------------------

def upload_dataset() -> str:
    """
    Upload the JSONL evaluation dataset to Azure AI Foundry
    and return its ID.
    """

    section("Step 1: Uploading evaluation dataset")

    dataset_path = (
        Path(__file__).parent.parent.parent
        / "data"
        / "trail_guide_evaluation_dataset.jsonl"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}.\n"
            "Make sure you are running the script from the repository root."
        )

    print(f"\nDataset: {dataset_path.name}")
    print("Uploading...")

    try:
        data_id = project_client.datasets.upload_file(
            name=dataset_name,
            version=dataset_version,
            file_path=str(dataset_path),
        ).id

        print("\n✓ Dataset uploaded successfully")

    except Exception as upload_error:

        # If this version was already uploaded in a previous run,
        # reuse it.
        if "already exists" in str(upload_error):

            print(
                f"\n  Dataset version {dataset_version} "
                "already exists in Foundry."
            )

            print("  Retrieving existing dataset ID...")

            dataset_obj = project_client.datasets.get(
                name=dataset_name,
                version=dataset_version,
            )

            data_id = dataset_obj.id

            print("  ✓ Using existing dataset")

        else:
            raise

    print(f"  Dataset ID: {data_id}")

    return data_id


# ---------------------------------------------------------------------------
# Step 2 – Create the evaluation definition
# ---------------------------------------------------------------------------

def create_evaluation_definition():
    """
    Register an evaluation definition in Foundry.

    Evaluators:
    - Intent Resolution
    - Relevance
    - Groundedness
    """

    section("Step 2: Creating evaluation definition")

    print("\nConfiguration:")
    print(f"  Judge Model: {model_deployment_name}")
    print(
        "  Evaluators: "
        "Intent Resolution, Relevance, Groundedness"
    )

    # Tell Foundry the shape of each record in the dataset
    data_source_config = DataSourceConfigCustom(
        type="custom",
        item_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string"
                },
                "response": {
                    "type": "string"
                },
                "ground_truth": {
                    "type": "string"
                },
            },
            "required": [
                "query",
                "response",
                "ground_truth",
            ],
        },
    )

    # Built-in evaluator configuration
    testing_criteria = [

        {
            "type": "azure_ai_evaluator",
            "name": "intent_resolution",
            "evaluator_name": "builtin.intent_resolution",
            "initialization_parameters": {
                "deployment_name": model_deployment_name
            },
            "data_mapping": {
                "query": "{{item.query}}",
                "response": "{{item.response}}",
            },
        },

        {
            "type": "azure_ai_evaluator",
            "name": "relevance",
            "evaluator_name": "builtin.relevance",
            "initialization_parameters": {
                "deployment_name": model_deployment_name
            },
            "data_mapping": {
                "query": "{{item.query}}",
                "response": "{{item.response}}",
            },
        },

        {
            "type": "azure_ai_evaluator",
            "name": "groundedness",
            "evaluator_name": "builtin.groundedness",
            "initialization_parameters": {
                "deployment_name": model_deployment_name
            },
            "data_mapping": {
                "query": "{{item.query}}",
                "response": "{{item.response}}",
                "context": "{{item.ground_truth}}",
            },
        },
    ]

    print("\nCreating evaluation...")

    eval_object = client.evals.create(
        name="Trail Guide Quality Evaluation",
        data_source_config=data_source_config,
        testing_criteria=testing_criteria,
    )

    print("\n✓ Evaluation definition created")
    print(f"  Evaluation ID: {eval_object.id}")

    return eval_object


# ---------------------------------------------------------------------------
# Step 3 – Start the evaluation run
# ---------------------------------------------------------------------------

def run_evaluation(eval_object, data_id):
    """
    Start a cloud evaluation run.
    """

    section("Step 3: Running cloud evaluation")

    eval_run = client.evals.runs.create(
        eval_id=eval_object.id,
        name="trail-guide-baseline-eval",
        data_source=CreateEvalJSONLRunDataSourceParam(
            type="jsonl",
            source=SourceFileID(
                type="file_id",
                id=data_id,
            ),
        ),
    )

    print("\n✓ Evaluation run started")
    print(f"  Run ID: {eval_run.id}")
    print(f"  Status: {eval_run.status}")

    print(
        "\nThis may take 15-60+ minutes for 89 items "
        "depending on capacity and quota..."
    )

    return eval_run


# ---------------------------------------------------------------------------
# Step 4 – Poll until the run finishes
# ---------------------------------------------------------------------------

def poll_for_results(eval_object, eval_run):
    """
    Repeatedly check the run status until it completes.
    """

    section("Step 4: Polling for completion")

    start_time = time.time()

    while True:

        run = client.evals.runs.retrieve(
            run_id=eval_run.id,
            eval_id=eval_object.id,
        )

        elapsed = int(time.time() - start_time)

        if run.status == "completed":

            print(
                f"\n\n✓ Evaluation completed successfully"
            )

            print(
                f"  Total time: {elapsed} seconds"
            )

            break

        elif run.status == "failed":

            error_detail = (
                getattr(run, "error", None)
                or "No additional details available."
            )

            raise RuntimeError(
                f"Evaluation run failed after {elapsed}s.\n"
                f"  Eval ID : {eval_object.id}\n"
                f"  Run ID  : {eval_run.id}\n"
                f"  Error   : {error_detail}\n"
                f"  To inspect: open Azure AI Foundry portal > Evaluations"
            )

        else:

            print(
                f"  [{elapsed}s] Status: {run.status}",
                end="\r",
                flush=True,
            )

            time.sleep(10)

    return run


# ---------------------------------------------------------------------------
# Step 5 – Collect scores and save results
# ---------------------------------------------------------------------------

def retrieve_and_display_results(eval_object, run):
    """
    Fetch per-item evaluator outputs, compute aggregate statistics,
    print a human-readable summary, and write the same summary
    to RESULTS_FILE.
    """

    section("Step 5: Retrieving results")

    print("\nEvaluation Summary")

    # Retrieve every scored item from the run
    output_items = list(
        client.evals.runs.output_items.list(
            run_id=run.id,
            eval_id=eval_object.id,
        )
    )

    print(
        f"  Retrieved output items: {len(output_items)}"
    )

    # ------------------------------------------------------------------
    # DEBUG
    # ------------------------------------------------------------------
    # This lets us see the actual structure returned by the installed
    # SDK version.
    #
    # Keep this initially. Once confirmed, it can be removed.
    # ------------------------------------------------------------------

    if output_items:
        print("\nDEBUG: First evaluation output item:")
        print(output_items[0])

    # ------------------------------------------------------------------
    # Separate errored and completed items
    # ------------------------------------------------------------------

    errored_items = [
        item
        for item in output_items
        if getattr(item, "status", None) == "error"
    ]

    scored_items = [
        item
        for item in output_items
        if getattr(item, "status", None) == "completed"
    ]

    if errored_items:

        print(
            f"\n  ⚠ {len(errored_items)} "
            "item(s) errored during evaluation."
        )

        print(
            "    First error: "
            f"{getattr(errored_items[0], 'error', 'details unavailable')}"
        )

        print(
            "    Open Azure AI Foundry portal > "
            "Evaluations to inspect all failed items."
        )

    # ------------------------------------------------------------------
    # Collect individual scores
    # ------------------------------------------------------------------

    scores: dict[str, list[float]] = {
        "intent_resolution": [],
        "relevance": [],
        "groundedness": [],
    }

    for item in scored_items:

        # Current Foundry SDK output items expose evaluator
        # results through the results collection.
        results = getattr(item, "results", None)

        if results is None:

            print(
                "\nDEBUG: No 'results' attribute found on item:"
            )

            print(item)

            continue

        for result in results:

            # Handle dictionary-style results
            if isinstance(result, dict):

                name = result.get("name")
                score = result.get("score")

            # Handle SDK model/object-style results
            else:

                name = getattr(
                    result,
                    "name",
                    None,
                )

                score = getattr(
                    result,
                    "score",
                    None,
                )

            if name in scores and score is not None:

                scores[name].append(
                    float(score)
                )

    # ------------------------------------------------------------------
    # Build summary
    # ------------------------------------------------------------------

    metric_labels = {
        "intent_resolution": "Intent Resolution",
        "relevance": "Relevance        ",
        "groundedness": "Groundedness     ",
    }

    lines = [
        "=" * 80,
        " Trail Guide Agent - Evaluation Results",
        "=" * 80,
        "",
        f"  Eval ID      : {eval_object.id}",
        f"  Run ID       : {run.id}",
        f"  Total items  : {len(output_items)}",
        f"  Errored items: {len(errored_items)}",
        f"  Scored items : {len(scored_items)}",
        "",
        "Average Scores (1-5 scale, threshold: 3)",
    ]

    any_scores = False

    pass_lines = [
        "",
        "Pass Rates (score >= 3)",
    ]

    for key, label in metric_labels.items():

        values = scores[key]

        if values:

            any_scores = True

            avg = (
                sum(values)
                / len(values)
            )

            pass_rate = (
                sum(
                    1
                    for value in values
                    if value >= 3
                )
                / len(values)
                * 100
            )

            lines.append(
                f"  {label}: "
                f"{avg:.2f} "
                f"(n={len(values)})"
            )

            pass_lines.append(
                f"  {label}: "
                f"{pass_rate:.1f}%"
            )

        else:

            lines.append(
                f"  {label}: No scores returned"
            )

            pass_lines.append(
                f"  {label}: No scores returned"
            )

    lines.extend(pass_lines)

    summary = "\n".join(lines)

    print("\n" + summary)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    RESULTS_FILE.write_text(
        summary,
        encoding="utf-8",
    )

    print(
        f"\n  Results saved to {RESULTS_FILE}"
    )

    # ------------------------------------------------------------------
    # Report URL
    # ------------------------------------------------------------------

    report_url = getattr(
        run,
        "report_url",
        None,
    )

    if not report_url:

        report_url = (
            f"{endpoint.rstrip('/')}"
            f"/evaluations/{eval_object.id}"
            f"/runs/{run.id}"
        )

    print(
        f"  Report URL: {report_url}"
    )

    # ------------------------------------------------------------------
    # GitHub Actions output
    # ------------------------------------------------------------------

    github_output = os.environ.get(
        "GITHUB_OUTPUT"
    )

    if github_output:

        with open(
            github_output,
            "a",
            encoding="utf-8",
        ) as gh_out:

            gh_out.write(
                f"report_url={report_url}\n"
            )

        print(
            "  GitHub Actions output set: "
            f"report_url={report_url}"
        )

    return output_items


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate the full evaluation pipeline.
    """

    section(
        " Trail Guide Agent - Cloud Evaluation"
    )

    print("\nConfiguration:")

    print(
        f"  Project: {endpoint}"
    )

    print(
        f"  Model: {model_deployment_name}"
    )

    print(
        f"  Dataset: "
        f"{dataset_name} "
        f"(v{dataset_version})"
    )

    try:

        # Step 1
        data_id = upload_dataset()

        # Step 2
        eval_object = create_evaluation_definition()

        # Step 3
        eval_run = run_evaluation(
            eval_object,
            data_id,
        )

        # Step 4
        run = poll_for_results(
            eval_object,
            eval_run,
        )

        # Step 5
        retrieve_and_display_results(
            eval_object,
            run,
        )

        section(
            "Cloud evaluation complete"
        )

        print("\nNext steps:")

        print(
            "  1. Review detailed results "
            "in Microsoft Foundry portal"
        )

        print(
            "  2. Analyze patterns in "
            "successful and failed evaluations"
        )

        print(
            f"  3. Commit {RESULTS_FILE} "
            "and push so the PR workflow can use it"
        )

    except Exception as e:

        error_message = (
            f"{'=' * 80}\n"
            f" Trail Guide Agent - Evaluation FAILED\n"
            f"{'=' * 80}\n"
            f"\nError: {e}\n"
            f"\nTroubleshooting:\n"
            f"  - Verify AZURE_AI_PROJECT_ENDPOINT in .env file\n"
            f"  - Check Azure credentials: az login\n"
            f"  - Ensure GPT-5.1 model is deployed and accessible\n"
            f"  - Ensure the caller has Foundry User access "
            f"at the AI account scope\n"
            f"  - If you just ran azd up, wait 1-2 minutes "
            f"for role propagation and retry once\n"
        )

        print(error_message)

        RESULTS_FILE.write_text(
            error_message,
            encoding="utf-8",
        )

        sys.exit(1)


if __name__ == "__main__":
    main()