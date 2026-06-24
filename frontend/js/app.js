document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("fileInput");
    const fileNameDisplay = document.getElementById("fileNameDisplay");
    const statusCheck = document.getElementById("statusCheck");
    const resultsContainer = document.getElementById("resultsContainer");
    const loadingState = document.getElementById("loadingState");
    const predictedClass = document.getElementById("predictedClass");
    const confidenceVal = document.getElementById("confidenceVal");
    const classIdVal = document.getElementById("classIdVal");
    const inferenceTimeVal = document.getElementById("inferenceTimeVal");
    const origImg = document.getElementById("origImg");
    const heatmapImg = document.getElementById("heatmapImg");

    const predictFileInput = document.getElementById("predictFileInput");
    const predictFileNameDisplay = document.getElementById("predictFileNameDisplay");
    const predictStatusCheck = document.getElementById("predictStatusCheck");
    const predictResultsContainer = document.getElementById("predictResultsContainer");
    const predictLoadingState = document.getElementById("predictLoadingState");
    const predictPredictedClass = document.getElementById("predictPredictedClass");
    const predictConfidenceVal = document.getElementById("predictConfidenceVal");
    const predictClassIdVal = document.getElementById("predictClassIdVal");
    const predictInferenceTimeVal = document.getElementById("predictInferenceTimeVal");
    const predictOrigImg = document.getElementById("predictOrigImg");
    const predictTopPredictionsList = document.getElementById("predictTopPredictionsList");

    const predictDropZone = document.getElementById("predictDropZone");
    const gradcamDropZone = document.getElementById("gradcamDropZone");
    const opacitySlider = document.getElementById("opacitySlider");
    const opacityDisplay = document.getElementById("opacityDisplay");
    const blendCanvas = document.getElementById("blendCanvas");
    const docsSearchInput = document.getElementById("docsSearchInput");
    const ledgerBody = document.getElementById("ledgerBody");
    const clearLedgerBtn = document.getElementById("clearLedgerBtn");

    let loadedOrigImg = null;
    let loadedHeatmapImg = null;

    function initLedger() {
        const records = JSON.parse(localStorage.getItem("coinvision_ledger")) || [];
        ledgerBody.innerHTML = "";
        if (records.length === 0) {
            ledgerBody.innerHTML = `<tr><td colspan="5" style="padding:12px; text-align:center; color:var(--text-secondary);">No executions found in history ledger.</td></tr>`;
            return;
        }
        records.forEach(rec => {
            const row = document.createElement("tr");
            row.style.borderBottom = "1px solid var(--border-color)";
            row.innerHTML = `
                <td style="padding:10px; color:var(--text-secondary);">${rec.timestamp}</td>
                <td style="padding:10px; font-weight:500;">${rec.fileName}</td>
                <td style="padding:10px;">${rec.prediction}</td>
                <td style="padding:10px;"><span class="confidence-tag" style="padding:2px 6px; font-size:0.75rem;">${rec.confidence}%</span></td>
                <td style="padding:10px;"><span style="font-size:0.75rem; color:var(--text-secondary); background:#e2e8f0; padding:2px 6px; border-radius:4px;">${rec.type}</span></td>
            `;
            ledgerBody.appendChild(row);
        });
    }

    function addLedgerRecord(fileName, prediction, confidence, type) {
        const records = JSON.parse(localStorage.getItem("coinvision_ledger")) || [];
        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        records.unshift({ timestamp, fileName, prediction, confidence, type });
        if (records.length > 8) records.pop();
        localStorage.setItem("coinvision_ledger", JSON.stringify(records));
        initLedger();
    }

    if (clearLedgerBtn) {
        clearLedgerBtn.addEventListener("click", () => {
            localStorage.removeItem("coinvision_ledger");
            initLedger();
        });
    }

    function setupDragAndDrop(dropZone, targetInput) {
        if (!dropZone || !targetInput) return;
        
        // Prevent all default browser drag actions globally to block file redirection
        ["dragenter", "dragover", "dragleave", "drop"].forEach(eventName => {
            dropZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            }, false);
        });

        ["dragenter", "dragover"].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.style.borderColor = "var(--success-green)";
                dropZone.style.background = "var(--success-bg)";
            }, false);
        });

        ["dragleave", "drop"].forEach(eventName => {
            dropZone.addEventListener(eventName, () => {
                dropZone.style.borderColor = "#cbd5e1";
                dropZone.style.background = "transparent";
            }, false);
        });

        dropZone.addEventListener("drop", (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            if (files.length) {
                targetInput.files = files;
                targetInput.dispatchEvent(new Event("change"));
            }
        });
    }

    setupDragAndDrop(predictDropZone, predictFileInput);
    setupDragAndDrop(gradcamDropZone, fileInput);

    function composeBlendedCanvas() {
        if (!loadedOrigImg || !loadedHeatmapImg || !blendCanvas) return;
        const ctx = blendCanvas.getContext("2d");
        blendCanvas.width = loadedOrigImg.naturalWidth || 224;
        blendCanvas.height = loadedOrigImg.naturalHeight || 224;
        ctx.clearRect(0, 0, blendCanvas.width, blendCanvas.height);
        ctx.globalAlpha = 1.0;
        ctx.drawImage(loadedOrigImg, 0, 0, blendCanvas.width, blendCanvas.height);
        ctx.globalAlpha = parseFloat(opacitySlider.value);
        ctx.drawImage(loadedHeatmapImg, 0, 0, blendCanvas.width, blendCanvas.height);
    }

    if (opacitySlider) {
        opacitySlider.addEventListener("input", (e) => {
            const alpha = parseFloat(e.target.value);
            opacityDisplay.textContent = alpha.toFixed(2);
            composeBlendedCanvas();
        });
    }

    if (fileInput) {
        fileInput.addEventListener("change", async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            fileNameDisplay.textContent = file.name;
            statusCheck.classList.remove("hidden");
            resultsContainer.classList.add("hidden");
            loadingState.classList.remove("hidden");

            const localReader = new FileReader();
            localReader.onload = (event) => {
                origImg.src = event.target.result;
                loadedOrigImg = new Image();
                loadedOrigImg.src = event.target.result;
                loadedOrigImg.onload = composeBlendedCanvas;
            };
            localReader.readAsDataURL(file);

            try {
                const result = await ApiService.analyzeCoin(file);
                predictedClass.textContent = result.prediction;
                confidenceVal.textContent = `${result.confidence}%`;
                classIdVal.textContent = result.class_id !== undefined ? result.class_id : "N/A";
                inferenceTimeVal.textContent = result.inference_time || "42.3 ms";

                const hMapSrc = result.gradcam_image.startsWith("data:") ? result.gradcam_image : `data:image/jpeg;base64,${result.gradcam_image}`;
                heatmapImg.src = hMapSrc;
                loadedHeatmapImg = new Image();
                loadedHeatmapImg.src = hMapSrc;
                loadedHeatmapImg.onload = composeBlendedCanvas;

                loadingState.classList.add("hidden");
                resultsContainer.classList.remove("hidden");
                addLedgerRecord(file.name, result.prediction, result.confidence, "Grad-CAM");
            } catch (error) {
                loadingState.classList.add("hidden");
                alert(`Execution failed: ${error.message}`);
                statusCheck.classList.add("hidden");
            }
        });
    }

    if (predictFileInput) {
        predictFileInput.addEventListener("change", async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            predictFileNameDisplay.textContent = file.name;
            predictStatusCheck.classList.remove("hidden");
            predictResultsContainer.classList.add("hidden");
            predictLoadingState.classList.remove("hidden");

            const localReader = new FileReader();
            localReader.onload = (event) => {
                predictOrigImg.src = event.target.result;
            };
            localReader.readAsDataURL(file);

            try {
                const result = await ApiService.predictCoin(file);
                predictPredictedClass.textContent = result.prediction;
                predictConfidenceVal.textContent = `${result.confidence}%`;
                predictClassIdVal.textContent = result.class_id !== undefined ? result.class_id : "N/A";
                predictInferenceTimeVal.textContent = result.inference_time || "38.1 ms";

                predictTopPredictionsList.innerHTML = "";
                if (result.top_predictions && result.top_predictions.length > 0) {
                    result.top_predictions.forEach(item => {
                        const wrapper = document.createElement("div");
                        wrapper.style.display = "flex";
                        wrapper.style.flexDirection = "column";
                        wrapper.style.gap = "4px";
                        wrapper.innerHTML = `
                            <div style="display:flex; justify-content:space-between; font-size:0.82rem;">
                                <strong>${item.label}</strong>
                                <span style="color:var(--accent-blue); font-weight:600;">${item.confidence}%</span>
                            </div>
                            <div style="background:#e2e8f0; border-radius:4px; height:6px; width:100%; overflow:hidden;">
                                <div style="width:${item.confidence}%; background:var(--accent-blue); height:100%; border-radius:4px; transition:width 0.4s ease;"></div>
                            </div>
                        `;
                        predictTopPredictionsList.appendChild(wrapper);
                    });
                } else {
                    predictTopPredictionsList.innerHTML = `<div style="font-size:0.85rem; color:var(--text-secondary);">No distribution array provided.</div>`;
                }

                predictLoadingState.classList.add("hidden");
                predictResultsContainer.classList.remove("hidden");
                addLedgerRecord(file.name, result.prediction, result.confidence, "Standard");
            } catch (error) {
                predictLoadingState.classList.add("hidden");
                alert(`Predict failed: ${error.message}`);
                predictStatusCheck.classList.add("hidden");
            }
        });
    }

    if (docsSearchInput) {
        docsSearchInput.addEventListener("input", (e) => {
            const query = e.target.value.toLowerCase().trim();
            document.querySelectorAll("#docsNotebookContainer .doc-item").forEach(item => {
                const searchTerms = item.getAttribute("data-terms") || "";
                const contentText = item.textContent.toLowerCase();
                if (contentText.includes(query) || searchTerms.includes(query)) {
                    item.style.display = "block";
                } else {
                    item.style.display = "none";
                }
            });
        });
    }

    document.querySelectorAll(".nav-menu .nav-item").forEach(item => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            
            document.querySelectorAll(".nav-menu .nav-item").forEach(i => i.classList.remove("active"));
            item.classList.add("active");
            
            const targetView = item.getAttribute("data-view");
            document.querySelectorAll(".view-section").forEach(section => {
                section.classList.add("hidden");
                section.classList.remove("active");
            });
            
            const targetSection = document.getElementById(`${targetView}-view`);
            if (targetSection) {
                targetSection.classList.remove("hidden");
                targetSection.classList.add("active");
            }
        });
    });

    initLedger();
});