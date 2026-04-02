const form = document.getElementById("upload-form");
const apiUrlInput = document.getElementById("api-url");
const csvFileInput = document.getElementById("csv-file");
const submitButton = document.getElementById("submit-button");
const statusBox = document.getElementById("status-box");
const resultsSection = document.getElementById("results-section");
const totalRowsElement = document.getElementById("total-rows");
const suspiciousRowsElement = document.getElementById("suspicious-rows");
const avgRiskScoreElement = document.getElementById("avg-risk-score");
const outputFilePathElement = document.getElementById("output-file-path");
const resultsTableBody = document.getElementById("results-table-body");


function setStatus(message, statusType) {
    statusBox.textContent = message;
    statusBox.className = `status-box ${statusType}`;
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


function renderFlagBadge(flagValue) {
    const isTrue = Number(flagValue) === 1;
    const badgeClass = isTrue ? "flag-badge flag-yes" : "flag-badge flag-no";
    const badgeText = isTrue ? "Yes" : "No";
    return `<span class="${badgeClass}">${badgeText}</span>`;
}


function renderResultsTable(records) {
    if (!records || records.length === 0) {
        resultsTableBody.innerHTML = `
            <tr>
                <td colspan="7">No suspicious rows were returned by the backend.</td>
            </tr>
        `;
        return;
    }

    const rowsHtml = records
        .map((record) => {
            return `
                <tr>
                    <td>${record.order_id ?? ""}</td>
                    <td>${record.customer_id ?? ""}</td>
                    <td>${formatDecimal(record.order_amount, 2)}</td>
                    <td>${formatDecimal(record.fraud_risk_score, 4)}</td>
                    <td>${renderFlagBadge(record.suspicious_flag)}</td>
                    <td>${renderFlagBadge(record.shipping_mismatch_flag)}</td>
                    <td>${renderFlagBadge(record.high_discount_flag)}</td>
                </tr>
            `;
        })
        .join("");

    resultsTableBody.innerHTML = rowsHtml;
}


async function submitCsvForPrediction(event) {
    event.preventDefault();

    const apiBaseUrl = apiUrlInput.value.trim().replace(/\/$/, "");
    const selectedFile = csvFileInput.files[0];

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

    submitButton.disabled = true;
    setStatus("Uploading CSV file and running fraud detection...", "status-loading");

    try {
        const response = await fetch(`${apiBaseUrl}/predict`, {
            method: "POST",
            body: formData,
        });

        const responseData = await response.json();

        if (!response.ok) {
            const errorMessage = responseData.detail || "Prediction failed.";
            throw new Error(errorMessage);
        }

        totalRowsElement.textContent = formatNumber(responseData.summary.total_rows);
        suspiciousRowsElement.textContent = formatNumber(responseData.summary.suspicious_rows);
        avgRiskScoreElement.textContent = formatDecimal(
            responseData.summary.average_suspicious_risk_score,
            4
        );
        const blobUrl = responseData.blob_upload?.blob_url;
        outputFilePathElement.textContent = blobUrl || responseData.output_file_path || "Not available";
        renderResultsTable(responseData.top_suspicious_records);

        resultsSection.classList.remove("hidden");
        setStatus("Prediction completed successfully.", "status-success");
    } catch (error) {
        setStatus(`Error: ${error.message}`, "status-error");
    } finally {
        submitButton.disabled = false;
    }
}


form.addEventListener("submit", submitCsvForPrediction);
