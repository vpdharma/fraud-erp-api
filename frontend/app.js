const form = document.getElementById("upload-form");
const apiUrlInput = document.getElementById("api-url");
const csvFileInput = document.getElementById("csv-file");
const modelSelect = document.getElementById("model-select");
const submitButton = document.getElementById("submit-button");
const statusBox = document.getElementById("status-box");
const resultsSection = document.getElementById("results-section");
const totalRowsElement = document.getElementById("total-rows");
const suspiciousRowsElement = document.getElementById("suspicious-rows");
const avgRiskScoreElement = document.getElementById("avg-risk-score");
const outputFilePathElement = document.getElementById("output-file-path");
const modelUsedInfoElement = document.getElementById("model-used-info");
const singleTableWrapper = document.getElementById("single-table-wrapper");
const resultsTableBody = document.getElementById("results-table-body");
const compareSection = document.getElementById("compare-section");
const benchmarkWrapper = document.getElementById("benchmark-wrapper");
const benchmarkNoteElement = document.getElementById("benchmark-note");
const benchmarkTableBody = document.getElementById("benchmark-table-body");
const chartContainer = document.getElementById("chart-container");
const perModelSummaryBody = document.getElementById("per-model-summary-body");
const consensusLineElement = document.getElementById("consensus-line");
const liveMetricsWrapper = document.getElementById("live-metrics-wrapper");
const liveMetricsNoteElement = document.getElementById("live-metrics-note");
const liveMetricsBody = document.getElementById("live-metrics-body");
const compareTableHead = document.getElementById("compare-table-head");
const compareTableBody = document.getElementById("compare-table-body");

const COMPARE_OPTION_VALUE = "__compare__";
const CHART_COLORS = ["#0e6b56", "#4a6fa5", "#b98a2e", "#b1492f"];

let cachedBenchmark = null;


function setStatus(message, statusType) {
    statusBox.textContent = message;
    statusBox.className = `status-box ${statusType}`;
}


function getApiBaseUrl() {
    return apiUrlInput.value.trim().replace(/\/$/, "");
}


function formatNumber(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "0";
    }
    return Number(value).toLocaleString();
}


function formatDecimal(value, digits = 4) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "0.0000";
    }
    return Number(value).toFixed(digits);
}


function formatMetric(value, digits = 4) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
        return "—";
    }
    return Number(value).toFixed(digits);
}


function clearChildren(element) {
    while (element.firstChild) {
        element.removeChild(element.firstChild);
    }
}


function createCell(text, tag = "td") {
    const cell = document.createElement(tag);
    cell.textContent = text;
    return cell;
}


function createFlagBadgeCell(flagValue) {
    const cell = document.createElement("td");
    const badge = document.createElement("span");
    const isTrue = Number(flagValue) === 1;
    badge.className = isTrue ? "flag-badge flag-yes" : "flag-badge flag-no";
    badge.textContent = isTrue ? "Yes" : "No";
    cell.appendChild(badge);
    return cell;
}


function renderEmptyRow(tableBody, columnCount, message) {
    clearChildren(tableBody);
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = columnCount;
    cell.textContent = message;
    row.appendChild(cell);
    tableBody.appendChild(row);
}


/* ------------------------------------------------------------------ */
/* Model list loading                                                  */
/* ------------------------------------------------------------------ */

async function loadModels() {
    const apiBaseUrl = getApiBaseUrl();
    if (!apiBaseUrl) {
        return;
    }

    try {
        const response = await fetch(`${apiBaseUrl}/models`);
        if (!response.ok) {
            return;
        }
        const data = await response.json();
        cachedBenchmark = data.benchmark || null;
        populateModelSelect(data.models || [], data.best_model);
    } catch (error) {
        // Backend may not be running yet; the default options still work.
        console.warn("Could not load model list:", error.message);
    }
}


function populateModelSelect(models, bestModel) {
    const previousValue = modelSelect.value;
    clearChildren(modelSelect);

    const autoOption = document.createElement("option");
    autoOption.value = "";
    const best = models.find((model) => model.name === bestModel);
    autoOption.textContent = best
        ? `Best single model (auto: ${best.display_name})`
        : "Best single model (auto)";
    modelSelect.appendChild(autoOption);

    models.forEach((model) => {
        const option = document.createElement("option");
        option.value = model.name;
        option.textContent = `${model.display_name} (${model.family})`;
        modelSelect.appendChild(option);
    });

    const compareOption = document.createElement("option");
    compareOption.value = COMPARE_OPTION_VALUE;
    compareOption.textContent = "Compare all models";
    modelSelect.appendChild(compareOption);

    const stillValid = Array.from(modelSelect.options).some(
        (option) => option.value === previousValue
    );
    modelSelect.value = stillValid ? previousValue : "";
}


/* ------------------------------------------------------------------ */
/* Single-model rendering                                              */
/* ------------------------------------------------------------------ */

function renderSummary(responseData) {
    totalRowsElement.textContent = formatNumber(responseData.summary.total_rows);
    suspiciousRowsElement.textContent = formatNumber(responseData.summary.suspicious_rows);
    avgRiskScoreElement.textContent = formatDecimal(
        responseData.summary.average_suspicious_risk_score,
        4
    );
    const blobUrl = responseData.blob_upload ? responseData.blob_upload.blob_url : null;
    outputFilePathElement.textContent =
        blobUrl || responseData.output_file_path || "Not available";
}


function renderResultsTable(records) {
    if (!records || records.length === 0) {
        renderEmptyRow(resultsTableBody, 7, "No suspicious rows were returned by the backend.");
        return;
    }

    clearChildren(resultsTableBody);
    records.forEach((record) => {
        const row = document.createElement("tr");
        row.appendChild(createCell(record.order_id ?? ""));
        row.appendChild(createCell(record.customer_id ?? ""));
        row.appendChild(createCell(formatDecimal(record.order_amount, 2)));
        row.appendChild(createCell(formatDecimal(record.fraud_risk_score, 4)));
        row.appendChild(createFlagBadgeCell(record.suspicious_flag));
        row.appendChild(createFlagBadgeCell(record.shipping_mismatch_flag));
        row.appendChild(createFlagBadgeCell(record.high_discount_flag));
        resultsTableBody.appendChild(row);
    });
}


function renderSinglePrediction(responseData) {
    renderSummary(responseData);

    modelUsedInfoElement.textContent =
        `Model used: ${responseData.model_display_name || responseData.model_used}` +
        (responseData.threshold_used !== null && responseData.threshold_used !== undefined
            ? ` (decision threshold ${formatDecimal(responseData.threshold_used, 4)})`
            : "");
    modelUsedInfoElement.classList.remove("hidden");

    renderResultsTable(responseData.top_suspicious_records);
    singleTableWrapper.classList.remove("hidden");
    compareSection.classList.add("hidden");
    resultsSection.classList.remove("hidden");
}


/* ------------------------------------------------------------------ */
/* Comparison rendering                                                */
/* ------------------------------------------------------------------ */

function effectiveMetrics(modelMetrics) {
    return modelMetrics.metrics_tuned || modelMetrics.metrics_default || {};
}


function renderBenchmark(benchmark) {
    if (!benchmark || !benchmark.models) {
        benchmarkWrapper.classList.add("hidden");
        return;
    }
    benchmarkWrapper.classList.remove("hidden");
    benchmarkNoteElement.textContent = benchmark.headline_note || "";

    clearChildren(benchmarkTableBody);
    Object.entries(benchmark.models).forEach(([name, modelMetrics]) => {
        const row = document.createElement("tr");
        const isBest = name === benchmark.best_model;

        const nameCell = createCell(
            (modelMetrics.display_name || name) + (isBest ? " ★" : "")
        );
        if (isBest) {
            nameCell.classList.add("best-model-cell");
        }
        row.appendChild(nameCell);
        row.appendChild(createCell(modelMetrics.family || ""));
        row.appendChild(createCell(formatMetric(modelMetrics.pr_auc)));
        row.appendChild(createCell(formatMetric(modelMetrics.roc_auc)));
        row.appendChild(
            createCell(
                modelMetrics.metrics_default
                    ? formatMetric(modelMetrics.metrics_default.recall)
                    : "—"
            )
        );
        row.appendChild(
            createCell(
                modelMetrics.metrics_tuned
                    ? formatMetric(modelMetrics.metrics_tuned.recall)
                    : "—"
            )
        );
        row.appendChild(
            createCell(
                modelMetrics.metrics_tuned
                    ? formatMetric(modelMetrics.metrics_tuned.precision)
                    : "—"
            )
        );
        row.appendChild(
            createCell(
                modelMetrics.metrics_tuned
                    ? formatMetric(modelMetrics.metrics_tuned.f1_score)
                    : "—"
            )
        );
        row.appendChild(
            createCell(
                modelMetrics.tuned_threshold !== null &&
                modelMetrics.tuned_threshold !== undefined
                    ? formatDecimal(modelMetrics.tuned_threshold, 4)
                    : "auto"
            )
        );
        benchmarkTableBody.appendChild(row);
    });

    renderBenchmarkChart(benchmark);
}


function renderBenchmarkChart(benchmark) {
    clearChildren(chartContainer);

    const modelEntries = Object.entries(benchmark.models);
    if (modelEntries.length === 0) {
        return;
    }

    const metricDefs = [
        { label: "PR-AUC", pick: (m) => m.pr_auc },
        { label: "ROC-AUC", pick: (m) => m.roc_auc },
        { label: "Recall", pick: (m) => effectiveMetrics(m).recall },
        { label: "F1", pick: (m) => effectiveMetrics(m).f1_score },
    ];

    const svgNamespace = "http://www.w3.org/2000/svg";
    const width = 760;
    const height = 300;
    const marginLeft = 44;
    const marginRight = 12;
    const marginTop = 34;
    const marginBottom = 44;
    const plotWidth = width - marginLeft - marginRight;
    const plotHeight = height - marginTop - marginBottom;

    const svg = document.createElementNS(svgNamespace, "svg");
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "Benchmark metric comparison chart");

    // Horizontal gridlines and y-axis labels at 0, 0.25, 0.5, 0.75, 1.
    for (let i = 0; i <= 4; i += 1) {
        const value = i / 4;
        const y = marginTop + plotHeight * (1 - value);
        const gridline = document.createElementNS(svgNamespace, "line");
        gridline.setAttribute("x1", marginLeft);
        gridline.setAttribute("x2", width - marginRight);
        gridline.setAttribute("y1", y);
        gridline.setAttribute("y2", y);
        gridline.setAttribute("stroke", "#d8d0c3");
        gridline.setAttribute("stroke-width", value === 0 ? "1.5" : "0.5");
        svg.appendChild(gridline);

        const label = document.createElementNS(svgNamespace, "text");
        label.setAttribute("x", marginLeft - 8);
        label.setAttribute("y", y + 4);
        label.setAttribute("text-anchor", "end");
        label.setAttribute("font-size", "11");
        label.setAttribute("fill", "#53645d");
        label.textContent = value.toFixed(2);
        svg.appendChild(label);
    }

    const groupWidth = plotWidth / modelEntries.length;
    const barGap = 4;
    const barWidth = Math.min(
        26,
        (groupWidth - 24 - barGap * (metricDefs.length - 1)) / metricDefs.length
    );

    modelEntries.forEach(([name, modelMetrics], groupIndex) => {
        const groupStart =
            marginLeft +
            groupWidth * groupIndex +
            (groupWidth - (barWidth * metricDefs.length + barGap * (metricDefs.length - 1))) / 2;

        metricDefs.forEach((metricDef, metricIndex) => {
            const rawValue = metricDef.pick(modelMetrics);
            const value = rawValue === null || rawValue === undefined ? 0 : Number(rawValue);
            const barHeight = plotHeight * value;
            const x = groupStart + metricIndex * (barWidth + barGap);
            const y = marginTop + plotHeight - barHeight;

            const bar = document.createElementNS(svgNamespace, "rect");
            bar.setAttribute("x", x);
            bar.setAttribute("y", y);
            bar.setAttribute("width", barWidth);
            bar.setAttribute("height", Math.max(barHeight, 0));
            bar.setAttribute("rx", "3");
            bar.setAttribute("fill", CHART_COLORS[metricIndex % CHART_COLORS.length]);

            const tooltip = document.createElementNS(svgNamespace, "title");
            tooltip.textContent =
                `${modelMetrics.display_name || name} — ${metricDef.label}: ` +
                formatMetric(rawValue);
            bar.appendChild(tooltip);
            svg.appendChild(bar);
        });

        const groupLabel = document.createElementNS(svgNamespace, "text");
        groupLabel.setAttribute("x", marginLeft + groupWidth * groupIndex + groupWidth / 2);
        groupLabel.setAttribute("y", height - marginBottom + 18);
        groupLabel.setAttribute("text-anchor", "middle");
        groupLabel.setAttribute("font-size", "12");
        groupLabel.setAttribute("fill", "#21302a");
        groupLabel.textContent = modelMetrics.display_name || name;
        svg.appendChild(groupLabel);
    });

    // Legend across the top.
    let legendX = marginLeft;
    metricDefs.forEach((metricDef, metricIndex) => {
        const swatch = document.createElementNS(svgNamespace, "rect");
        swatch.setAttribute("x", legendX);
        swatch.setAttribute("y", 10);
        swatch.setAttribute("width", 12);
        swatch.setAttribute("height", 12);
        swatch.setAttribute("rx", "2");
        swatch.setAttribute("fill", CHART_COLORS[metricIndex % CHART_COLORS.length]);
        svg.appendChild(swatch);

        const legendLabel = document.createElementNS(svgNamespace, "text");
        legendLabel.setAttribute("x", legendX + 17);
        legendLabel.setAttribute("y", 21);
        legendLabel.setAttribute("font-size", "12");
        legendLabel.setAttribute("fill", "#21302a");
        legendLabel.textContent = metricDef.label;
        svg.appendChild(legendLabel);

        legendX += 17 + metricDef.label.length * 7 + 22;
    });

    chartContainer.appendChild(svg);
}


function renderPerModelSummary(perModelSummary) {
    clearChildren(perModelSummaryBody);
    Object.values(perModelSummary).forEach((summary) => {
        const row = document.createElement("tr");
        row.appendChild(createCell(summary.display_name));
        row.appendChild(
            createCell(
                summary.tuned_threshold !== null && summary.tuned_threshold !== undefined
                    ? formatDecimal(summary.tuned_threshold, 4)
                    : "auto (contamination)"
            )
        );
        row.appendChild(createCell(formatNumber(summary.suspicious_rows)));
        row.appendChild(createCell(formatDecimal(summary.average_flagged_risk_score, 4)));
        perModelSummaryBody.appendChild(row);
    });
}


function renderConsensus(consensus) {
    consensusLineElement.textContent =
        `Consensus across ${consensus.models_total} models: ` +
        `${formatNumber(consensus.rows_flagged_by_any)} rows flagged by at least one model, ` +
        `${formatNumber(consensus.rows_flagged_by_majority)} by a majority ` +
        `(${consensus.majority_needed}+), and ` +
        `${formatNumber(consensus.rows_flagged_by_all)} by all models.`;
}


function renderLiveMetrics(liveMetrics, perModelSummary) {
    if (!liveMetrics) {
        liveMetricsWrapper.classList.add("hidden");
        return;
    }
    liveMetricsWrapper.classList.remove("hidden");
    liveMetricsNoteElement.textContent = liveMetrics.note || "";

    clearChildren(liveMetricsBody);
    Object.entries(liveMetrics.per_model).forEach(([name, metrics]) => {
        const displayName =
            (perModelSummary[name] && perModelSummary[name].display_name) || name;
        const row = document.createElement("tr");
        row.appendChild(createCell(displayName));
        row.appendChild(createCell(formatMetric(metrics.accuracy)));
        row.appendChild(createCell(formatMetric(metrics.precision)));
        row.appendChild(createCell(formatMetric(metrics.recall)));
        row.appendChild(createCell(formatMetric(metrics.f1_score)));
        row.appendChild(createCell(formatMetric(metrics.roc_auc)));
        row.appendChild(createCell(formatMetric(metrics.pr_auc)));
        liveMetricsBody.appendChild(row);
    });
}


function renderCompareTable(records, modelsCompared, perModelSummary) {
    clearChildren(compareTableHead);
    const headRow = document.createElement("tr");
    headRow.appendChild(createCell("Order ID", "th"));
    headRow.appendChild(createCell("Customer ID", "th"));
    headRow.appendChild(createCell("Order Amount", "th"));
    headRow.appendChild(createCell("Avg Risk", "th"));
    headRow.appendChild(createCell("Models Flagging", "th"));
    modelsCompared.forEach((name) => {
        const displayName =
            (perModelSummary[name] && perModelSummary[name].display_name) || name;
        headRow.appendChild(createCell(displayName, "th"));
    });
    compareTableHead.appendChild(headRow);

    const columnCount = 5 + modelsCompared.length;
    if (!records || records.length === 0) {
        renderEmptyRow(compareTableBody, columnCount, "No records were returned by the backend.");
        return;
    }

    clearChildren(compareTableBody);
    records.forEach((record) => {
        const row = document.createElement("tr");
        row.appendChild(createCell(record.order_id ?? ""));
        row.appendChild(createCell(record.customer_id ?? ""));
        row.appendChild(createCell(formatDecimal(record.order_amount, 2)));
        row.appendChild(createCell(formatDecimal(record.avg_risk_score, 4)));
        row.appendChild(
            createCell(`${formatNumber(record.consensus_count)}/${modelsCompared.length}`)
        );
        modelsCompared.forEach((name) => {
            row.appendChild(createFlagBadgeCell(record[`${name}_suspicious_flag`]));
        });
        compareTableBody.appendChild(row);
    });
}


function renderComparison(responseData) {
    renderSummary(responseData);

    const bestSummary = responseData.per_model_summary[responseData.best_model];
    modelUsedInfoElement.textContent =
        `Compared ${responseData.models_compared.length} models. ` +
        `Summary cards use the best model: ${bestSummary ? bestSummary.display_name : responseData.best_model}.`;
    modelUsedInfoElement.classList.remove("hidden");

    singleTableWrapper.classList.add("hidden");
    renderBenchmark(responseData.benchmark || cachedBenchmark);
    renderPerModelSummary(responseData.per_model_summary);
    renderConsensus(responseData.consensus);
    renderLiveMetrics(responseData.live_metrics, responseData.per_model_summary);
    renderCompareTable(
        responseData.top_suspicious_records,
        responseData.models_compared,
        responseData.per_model_summary
    );

    resultsSection.classList.remove("hidden");
    compareSection.classList.remove("hidden");
}


/* ------------------------------------------------------------------ */
/* Form submission                                                     */
/* ------------------------------------------------------------------ */

async function submitCsvForPrediction(event) {
    event.preventDefault();

    const apiBaseUrl = getApiBaseUrl();
    const selectedFile = csvFileInput.files[0];
    const selectedMode = modelSelect.value;

    if (!apiBaseUrl) {
        setStatus("Please enter the backend API URL.", "status-error");
        return;
    }

    if (!selectedFile) {
        setStatus("Please choose a CSV file before clicking the button.", "status-error");
        return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    const isCompare = selectedMode === COMPARE_OPTION_VALUE;
    let requestUrl = `${apiBaseUrl}/predict`;
    if (isCompare) {
        requestUrl = `${apiBaseUrl}/compare`;
    } else if (selectedMode) {
        requestUrl = `${apiBaseUrl}/predict?model=${encodeURIComponent(selectedMode)}`;
    }

    submitButton.disabled = true;
    setStatus(
        isCompare
            ? "Uploading CSV file and comparing all models..."
            : "Uploading CSV file and running fraud detection...",
        "status-loading"
    );

    try {
        const response = await fetch(requestUrl, {
            method: "POST",
            body: formData,
        });

        const responseData = await response.json();

        if (!response.ok) {
            const errorMessage = responseData.detail || "Prediction failed.";
            throw new Error(errorMessage);
        }

        if (isCompare) {
            renderComparison(responseData);
            setStatus("Model comparison completed successfully.", "status-success");
        } else {
            renderSinglePrediction(responseData);
            setStatus("Prediction completed successfully.", "status-success");
        }
    } catch (error) {
        setStatus(`Error: ${error.message}`, "status-error");
    } finally {
        submitButton.disabled = false;
    }
}


form.addEventListener("submit", submitCsvForPrediction);
apiUrlInput.addEventListener("change", loadModels);
document.addEventListener("DOMContentLoaded", loadModels);
