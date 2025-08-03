// Global variables
let selectedFile = null;
let analysisResult = null;

// DOM Elements - will be initialized after DOM loads
let uploadArea, imageInput, analyzeBtn, streamBtn;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    // Initialize DOM elements after DOM is loaded
    uploadArea = document.getElementById('uploadArea');
    imageInput = document.getElementById('imageInput');
    analyzeBtn = document.getElementById('analyzeBtn');
    streamBtn = document.getElementById('streamBtn');
    visualizeBtn = document.getElementById('visualizeBtn');
    
    // Ensure buttons start disabled
    disableButtons();
    
    // Initialize event listeners
    initializeEventListeners();
});

// Initialize all event listeners
function initializeEventListeners() {
    // File input handling
    if (imageInput) {
        imageInput.addEventListener('change', handleFileSelect);
    }
    
    // Drag and drop handling
    if (uploadArea) {
        uploadArea.addEventListener('dragover', handleDragOver);
        uploadArea.addEventListener('dragleave', handleDragLeave);
        uploadArea.addEventListener('drop', handleDrop);
        uploadArea.addEventListener('click', () => imageInput.click());
    }
}

// File selection handler
function handleFileSelect(e) {
    selectedFile = e.target.files[0];
    if (selectedFile) {
        updateUploadArea();
        enableButtons();
    }
}

// Drag and drop handlers
function handleDragOver(e) {
    e.preventDefault();
    if (uploadArea) {
        uploadArea.classList.add('dragover');
    }
}

function handleDragLeave(e) {
    e.preventDefault();
    if (uploadArea) {
        uploadArea.classList.remove('dragover');
    }
}

function handleDrop(e) {
    e.preventDefault();
    if (uploadArea) {
        uploadArea.classList.remove('dragover');
    }
    
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('image/')) {
        selectedFile = files[0];
        // Don't set imageInput.files as it triggers handleFileSelect
        updateUploadArea();
        enableButtons();
    }
}

// Update upload area display
function updateUploadArea() {
    if (!uploadArea) return;
    
    const uploadContent = uploadArea.querySelector('.upload-content');
    if (uploadContent) {
        uploadContent.innerHTML = `
            <div class="upload-icon">✅</div>
            <p><strong>${selectedFile.name}</strong></p>
            <p class="upload-hint">File selected successfully</p>
        `;
    }
}

// Enable analysis buttons
function enableButtons() {
    if (analyzeBtn) {
        analyzeBtn.disabled = false;
    }
    if (streamBtn) {
        streamBtn.disabled = false;
    }
    if (visualizeBtn) {
        visualizeBtn.disabled = false;
    }
}

// Disable analysis buttons
function disableButtons() {
    if (analyzeBtn) {
        analyzeBtn.disabled = true;
    }
    if (streamBtn) {
        streamBtn.disabled = true;
    }
    if (visualizeBtn) {
        visualizeBtn.disabled = true;
    }
}

// Regular analysis function (Custom YOLO + Gemini Flash Lite)
async function analyzeImage() {
    if (!selectedFile) return;

    console.log('Starting regular analysis with Custom YOLO + Gemini Flash Lite');
    
    // Show loading
    showLoading('Analyzing with Custom YOLO + Gemini Flash Lite...');
    hideResults();

    // Prepare form data
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('model', 'gemini-flash-lite');
    console.log('Sending request to /analyze-vision-custom with Gemini Flash Lite');

    try {
        // Use custom YOLO endpoint with Gemini Flash Lite
        const response = await fetch('/analyze-vision-custom', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
        }

        analysisResult = await response.json();
        console.log('Analysis result received:', analysisResult);
        
        // Hide loading and show results
        hideLoading();
        console.log('Calling displayVisionResults...');
        displayVisionResults(analysisResult);
        
    } catch (error) {
        console.error('Error analyzing image:', error);
        showError(`❌ Error: ${error.message}`);
    }
}

// Streaming analysis function (Custom YOLO + Gemini Flash Lite)
async function analyzeImageStream() {
    console.log('analyzeImageStream() called');
    if (!selectedFile) {
        console.log('No file selected for streaming');
        return;
    }

    console.log('Starting streaming analysis with Custom YOLO + Gemini Flash Lite');
    
    // Show loading with streaming indicator
    showLoading('Streaming analysis with Custom YOLO + Gemini Flash Lite...');
    hideResults();
    
    // Clear any previous results from results section
    clearPreviousResults();

    // Prepare form data
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('model', 'gemini-flash-lite');
    console.log('Streaming FormData prepared with Gemini Flash Lite');

    try {
        // Use custom YOLO streaming endpoint with Gemini Flash Lite
        const response = await fetch('/analyze-vision-custom-stream', {
            method: 'POST',
            body: formData
        });
        console.log('Streaming response received:', response.status, response.statusText);

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        await handleStreamingResponse(response, 'gemini-flash-lite');

    } catch (error) {
        console.error('Error streaming analysis:', error);
        showError(`❌ Streaming Error: ${error.message}`);
    }
}

// Handle streaming response
async function handleStreamingResponse(response, selectedModel) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const streamContentDiv = document.getElementById('streamContent');
    let streamedContent = '';
    let partialJson = '';
    let analysisData = null;
    const streamStartTime = Date.now();
    console.log(`🚀 Starting streaming analysis with ${selectedModel} at ${new Date().toLocaleTimeString()}`);

    while (true) {
        const { done, value } = await reader.read();
        if (done) {
            // Stream ended, do final parsing
            console.log('Stream ended, doing final parsing');
            await handleStreamCompletion(streamedContent, analysisData, streamStartTime, selectedModel);
            break;
        }

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
            if (line.startsWith('data: ')) {
                try {
                    const data = JSON.parse(line.slice(6));
                    
                    // Handle parallel streaming format
                    if (data.type === "shelf_completed") {
                        console.log(`✅ Shelf ${data.shelf_id} completed (${data.completed_count}/${data.total_shelves})`);
                        console.log(`⏱️  Shelf processing time: ${data.shelf_time_ms}ms`);
                        
                        // Store the shelf result for final combination
                        if (data.shelf_result) {
                            if (!analysisData) analysisData = { shelves: [] };
                            analysisData.shelves.push(data.shelf_result);
                        }
                        
                        // Show progressive results
                        if (analysisData) {
                            displayProgressiveResults(analysisData, streamContentDiv);
                        }
                        
                    } else if (data.type === "fallback") {
                        // Fallback to regular analysis
                        console.log('🔄 ' + data.message);
                        
                    } else if (data.type === "analysis_complete") {
                        // Final result from parallel streaming
                        if (data.error) {
                            console.error('❌ Parallel analysis failed:', data.error);
                            throw new Error(data.error);
                        }
                        console.log('🎉 Parallel analysis complete!');
                        analysisData = data.result;
                        streamedContent = JSON.stringify(data.result);
                        
                    } else if (data.chunk) {
                        // Regular streaming format
                        streamedContent += data.chunk;
                        partialJson += data.chunk;
                        
                        // Try to parse partial shelf data for progressive display
                        await handlePartialJson(partialJson, streamContentDiv, analysisData, streamStartTime);
                        
                    } else if (data.error) {
                        throw new Error(data.error);
                    }
                } catch (e) {
                    console.log('Non-JSON chunk:', line);
                }
            }
        }
    }
}

// Handle partial JSON during streaming
async function handlePartialJson(partialJson, streamContentDiv, analysisData, streamStartTime) {
    try {
        console.log('Attempting to parse JSON:', partialJson.substring(0, 200) + '...');
        
        // Clean the JSON by removing markdown formatting
        let cleanJson = partialJson;
        if (cleanJson.includes('```json')) {
            cleanJson = cleanJson.replace(/```json\s*/, '').replace(/```\s*$/, '');
        }
        if (cleanJson.includes('```')) {
            cleanJson = cleanJson.replace(/```\s*/, '');
        }
        
        // Check if JSON appears to be complete (has closing brace)
        const hasClosingBrace = (cleanJson.match(/{/g) || []).length === (cleanJson.match(/}/g) || []).length;
        const hasOpeningBrace = cleanJson.includes('{');
        
        // Try to parse the JSON if it looks complete enough
        if (hasOpeningBrace && cleanJson.includes('"shelves"')) {
            try {
                const parsedData = JSON.parse(cleanJson);
                if (parsedData.shelves && parsedData.shelves.length > 0) {
                    console.log(`📊 Progressive update: ${parsedData.shelves.length} shelves detected`);
                    displayProgressiveResults(parsedData, streamContentDiv);
                }
            } catch (parseError) {
                // JSON is incomplete, that's expected during streaming
            }
        }
        
        // Try to extract partial shelf data for progressive display (new format)
        const shelfMatches = cleanJson.match(/"shelf_position":\s*(\d+)[^}]*"products":\s*\[([^\]]*)\]/g);
        if (shelfMatches && shelfMatches.length > 0) {
            const currentTime = Date.now();
            const shelfCount = shelfMatches.length;
            console.log(`🕒 Shelf ${shelfCount} detected at ${new Date().toLocaleTimeString()} (${currentTime - streamStartTime}ms elapsed)`);
            
            console.log('Found partial shelf data, showing progressive results');
            const partialData = {
                is_display_detected: true,
                shelves: shelfMatches.map((match, index) => {
                    const positionMatch = match.match(/"shelf_position":\s*(\d+)/);
                    const goodsMatch = match.match(/"goods":\s*\[([^\]]*)\]/);
                    return {
                        shelf_position: positionMatch ? parseInt(positionMatch[1]) : index + 1,
                        shelf_visibility: 95,
                        goods: goodsMatch ? parsePartialProducts(goodsMatch[1]) : [],
                        notes: "Analysis in progress..."
                    };
                }),
                scores: {
                    product_filling: { value: 0, comment: "Analysis in progress..." },
                    product_neatness: { value: 0, comment: "Analysis in progress..." },
                    shelf_arrangement: { value: 0, comment: "Analysis in progress..." },
                    empty_space_detection: { value: 0, comment: "Analysis in progress..." },
                    overall_score: { value: 0, comment: "Analysis in progress..." }
                },
                total_score: 0,
                general_comment: "Analysis in progress..."
            };
            displayProgressiveResults(partialData, streamContentDiv);
        } else if (cleanJson.length > 100 && hasOpeningBrace && hasClosingBrace && (cleanJson.includes('"shelves"') || cleanJson.includes('"is_display_detected"'))) {
            // Try full JSON parse if we have enough data and braces are balanced
            console.log('Cleaned JSON:', cleanJson.substring(0, 200) + '...');
            const parsed = JSON.parse(cleanJson);
            console.log('JSON parsed successfully:', parsed);
            analysisData = parsed;
            // Handle the nested JSON structure
            const displayData = parsed.json || parsed;
            displayProgressiveResults(displayData, streamContentDiv);
        } else {
            // Show loading state for incomplete JSON
            showLoadingState(streamContentDiv);
        }
    } catch (e) {
        console.log('JSON parsing failed:', e.message);
        
        // Check if we have any meaningful partial data to show
        if (analysisData && Object.keys(analysisData).length > 0) {
            // Only show partial data if it's not just an error object
            if (!analysisData.error && !analysisData.shelves || analysisData.shelves.length > 0) {
                displayProgressiveResults(analysisData, streamContentDiv);
            } else {
                // Show loading state instead of error
                showLoadingState(streamContentDiv);
            }
        } else {
            // Show loading state for incomplete JSON
            showLoadingState(streamContentDiv);
        }
    }
}

// Handle stream completion
async function handleStreamCompletion(streamedContent, analysisData, streamStartTime, selectedModel) {
    try {
        let finalContent = streamedContent;
        // Clean the JSON by removing markdown formatting
        if (finalContent.includes('```json')) {
            finalContent = finalContent.replace(/```json\s*/, '').replace(/```\s*$/, '');
        }
        if (finalContent.includes('```')) {
            finalContent = finalContent.replace(/```\s*/, '');
        }
        
        let finalData;
        if (analysisData) {
            finalData = analysisData;
        } else {
            // Try to parse the final content, but handle incomplete JSON gracefully
            try {
                // Check if the content looks like valid JSON
                const trimmedContent = finalContent.trim();
                if (trimmedContent.length > 0 && 
                    (trimmedContent.startsWith('{') || trimmedContent.startsWith('['))) {
                    finalData = JSON.parse(trimmedContent);
                } else {
                    console.warn('Final content does not appear to be valid JSON:', trimmedContent);
                    // Create a fallback result
                    finalData = {
                        is_display_detected: false,
                        shelves: [],
                        scores: {
                            product_filling: { value: 0, comment: "Analysis incomplete" },
                            product_neatness: { value: 0, comment: "Analysis incomplete" },
                            shelf_arrangement: { value: 0, comment: "Analysis incomplete" },
                            empty_space_detection: { value: 0, comment: "Analysis incomplete" },
                            overall_score: { value: 0, comment: "Analysis incomplete" }
                        },
                        total_score: 0,
                        general_comment: "Analysis incomplete - unable to parse response"
                    };
                }
            } catch (parseError) {
                console.error('JSON parsing failed:', parseError);
                console.log('Raw content that failed to parse:', finalContent);
                // Create a fallback result
                finalData = {
                    is_display_detected: false,
                    shelves: [],
                    scores: {
                        product_filling: { value: 0, comment: "Analysis incomplete" },
                        product_neatness: { value: 0, comment: "Analysis incomplete" },
                        shelf_arrangement: { value: 0, comment: "Analysis incomplete" },
                        empty_space_detection: { value: 0, comment: "Analysis incomplete" },
                        overall_score: { value: 0, comment: "Analysis incomplete" }
                    },
                    total_score: 0,
                    general_comment: "Analysis incomplete - JSON parsing failed"
                };
            }
        }
        
        // Check if we have an error in the final data
        if (finalData.error) {
            console.error('Analysis failed:', finalData.error);
            showError(`Analysis failed: ${finalData.error}`);
            return;
        }
        
        // Handle different response formats
        if (finalData.type === "analysis_complete") {
            // Parallel streaming format
            analysisResult = finalData.result;
        } else if (finalData.json) {
            // Regular streaming format
            analysisResult = finalData.json;
        } else {
            // Direct result format
            analysisResult = finalData;
        }
        
        // Store the result globally
        window.analysisResult = analysisResult;
        
        // Log final results for model comparison
        const totalTime = Date.now() - streamStartTime;
        const finalScore = analysisResult.scores?.overall_score?.value || 'N/A';
        const shelfCount = analysisResult.shelves?.length || 0;
        console.log(`🏁 ${selectedModel} Analysis Complete:`);
        console.log(`   📊 Overall Score: ${finalScore}/100`);
        console.log(`   📦 Shelves Analyzed: ${shelfCount}`);
        console.log(`   ⏱️  Total Time: ${totalTime}ms`);
        console.log(`   🕒 Average per shelf: ${shelfCount > 0 ? Math.round(totalTime / shelfCount) : 0}ms`);
        
        // Move progressive content to results section before hiding loading
        const streamContentDiv = document.getElementById('streamContent');
        const resultsSection = document.getElementById('resultsSection');
        if (streamContentDiv && resultsSection) {
            // Move the progressive content to results section
            resultsSection.appendChild(streamContentDiv);
            
            // Update the progressive display with complete data
            displayProgressiveResults(analysisResult, streamContentDiv);
        }
        
        hideLoading();
        displayVisionResults(analysisResult);
    } catch (e) {
        console.error('Final parsing failed:', e);
        console.log('Streamed content that caused the error:', streamedContent);
        
        // Create a fallback result and show it
        const fallbackResult = {
            shelves: [],
            scores: {
                overall_score: { value: 0, comment: "Analysis failed" }
            },
            error: "Streaming analysis failed: " + e.message
        };
        
        // Try to display the fallback result
        try {
            hideLoading();
            displayVisionResults(fallbackResult);
        } catch (displayError) {
            console.error('Failed to display fallback result:', displayError);
            showRawContent(streamedContent);
        }
    }
}

// Parse partial products from JSON
function parsePartialProducts(productsJson) {
    try {
        // Extract product names from partial JSON
        const nameMatches = productsJson.match(/"name":\s*"([^"]+)"/g);
        if (nameMatches) {
            return nameMatches.map(match => {
                const name = match.match(/"name":\s*"([^"]+)"/)[1];
                return {
                    name: name,
                    full_name: name,
                    count: 1,
                    confidence: 0.9
                };
            });
        }
    } catch (e) {
        console.log('Error parsing partial products:', e);
    }
    return [];
}

// Display progressive results
function displayProgressiveResults(data, container) {
    console.log('displayProgressiveResults called with data:', data);
    
    // Clear container only on first call (when no content exists)
    if (!container.querySelector('.progressive-results')) {
        container.innerHTML = '';
    }
    
    let html = '<div class="progressive-results">';
    html += '<h4>🎯 Progressive Analysis Results</h4>';
    
    // Check if display was detected (new structure uses is_display_detected, assume true if shelves exist)
    const isDisplayDetected = data.is_display_detected !== false || (data.shelves && data.shelves.length > 0);
    if (!isDisplayDetected) {
        html += '<div style="background: #ffebee; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #d32f2f;">';
        html += '<h5>❌ No Store Display Detected</h5>';
        html += '<p>This image does not appear to show a store display or product stand.</p>';
        html += '</div>';
    } else {
        
        // Show shelves if available
        if (data.shelves && data.shelves.length > 0) {
            html += '<div style="margin-bottom: 20px;">';
            html += '<h5>📦 Shelf Analysis</h5>';
            data.shelves.forEach((shelf, index) => {
                html += `<div style="background: white; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #667eea;">`;
                html += `<h6>Shelf Level ${shelf.shelf_position || index + 1}</h6>`;
                html += `<p><strong>Visibility:</strong> ${shelf.shelf_visibility || 'N/A'}%</p>`;
                
                // Use new format (products)
                const products = shelf.products || [];
                if (products.length > 0) {
                    html += '<div style="margin-top: 10px;"><strong>Products Found:</strong><ul>';
                    products.forEach(product => {
                        const confidence = product.confidence ? Math.round(product.confidence * 100) : 'N/A';
                        html += `<li>${product.name} (${product.count} items${confidence !== 'N/A' ? `, ${confidence}% confidence` : ''})</li>`;
                    });
                    html += '</ul></div>';
                }
                
                if (shelf.notes) {
                    html += `<p style="color: #666; font-style: italic;">💡 ${shelf.notes}</p>`;
                }
                html += '</div>';
            });
            html += '</div>';
        }
    }
    

    
    // Show scores if available (handle both old and new format)
    const scores = data.scores;
    if (scores) {
        html += '<div style="margin-bottom: 20px;">';
        html += '<h5>📊 Analysis Scores</h5>';
        
        // Show overall score if available
        if (scores.overall_score) {
            html += `<div style="background: #e8f5e8; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #28a745; text-align: center;">`;
            html += `<div style="font-size: 24px;">🏆</div>`;
            html += `<div style="font-size: 24px; font-weight: bold; color: #28a745;">${scores.overall_score.value}/100</div>`;
            html += `<div style="font-size: 12px; color: #666;">Overall Score</div>`;
            if (scores.overall_score.comment) {
                html += `<div style="font-size: 11px; color: #666; margin-top: 5px;">${scores.overall_score.comment}</div>`;
            }
            html += '</div>';
        }
        
        html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">';
        
        // New format score items
        const newScoreItems = [
            { key: 'bannerVisibility', label: 'Banner Visibility', icon: '🎯' },
            { key: 'product_filling', label: 'Product Filling', icon: '📦' },
            { key: 'product_neatness', label: 'Product Neatness', icon: '✨' },
            { key: 'shelf_arrangement', label: 'Shelf Arrangement', icon: '🎯' },
            { key: 'empty_space_detection', label: 'Empty Space Detection', icon: '🔍' },
            { key: 'overall_score', label: 'Overall Score', icon: '⭐' }
        ];
        
        // Old format score items (fallback)
        const oldScoreItems = [
            { key: 'product_filling', label: 'Product Filling', icon: '📦' },
            { key: 'product_neatness', label: 'Product Neatness', icon: '✨' },
            { key: 'shelf_arrangement', label: 'Shelf Arrangement', icon: '🎯' },
            { key: 'empty_space_detection', label: 'Empty Space Detection', icon: '🔍' },
            { key: 'overall_score', label: 'Overall Score', icon: '🏆' }
        ];
        
        // Use new format (AnalysisScores structure)
        const scoreItems = oldScoreItems;
        
        scoreItems.forEach(item => {
            const score = scores[item.key];
            if (score && score.value !== undefined) {
                const color = score.value >= 80 ? '#d4edda' : score.value >= 60 ? '#fff3cd' : '#f8d7da';
                const borderColor = score.value >= 80 ? '#28a745' : score.value >= 60 ? '#ffc107' : '#dc3545';
                
                html += `<div style="background: ${color}; padding: 15px; border-radius: 8px; border-left: 4px solid ${borderColor}; text-align: center;">`;
                html += `<div style="font-size: 24px;">${item.icon}</div>`;
                html += `<div style="font-size: 24px; font-weight: bold; color: ${borderColor};">${score.value}/100</div>`;
                html += `<div style="font-size: 12px; color: #666;">${item.label}</div>`;
                if (score.comment) {
                    html += `<div style="font-size: 11px; color: #666; margin-top: 5px;">${score.comment}</div>`;
                }
                html += '</div>';
            }
        });
        html += '</div></div>';
    }
    
    // Show processing status
    html += '<div style="background: #e7f3ff; padding: 10px; border-radius: 5px; border-left: 4px solid #007bff;">';
    html += '<strong>⏳ Analysis in Progress...</strong> More details coming as the model processes the image.';
    html += '</div>';
    
    html += '</div>';
    
    // Replace the progressive results div if it exists, otherwise append
    const existingResults = container.querySelector('.progressive-results');
    if (existingResults) {
        existingResults.outerHTML = html;
    } else {
        container.innerHTML = html;
    }
}

// Display vision results
function displayVisionResults(analysisResult) {
    console.log('displayVisionResults called with:', analysisResult);
    
    // Use passed parameter or fall back to global result
    const resultToUse = analysisResult || window.analysisResult;
    
    // Store the result globally for other functions to use
    window.analysisResult = resultToUse;
    
    // Show results section
    const resultsSection = document.getElementById('resultsSection');
    console.log('Results section element:', resultsSection);
    
    if (!resultsSection) {
        console.error('Results section element not found!');
        showError('Results section not found. Please refresh the page.');
        return;
    }
    
    resultsSection.style.display = 'block';

    // Restore the original results structure for analysis display
    resultsSection.innerHTML = `
        <!-- Stats Grid -->
        <div class="stats-section">
            <h2>📊 Analysis Statistics</h2>
            <div id="statsGrid" class="stats-grid"></div>
        </div>

        <!-- Image Analysis Section -->
        <div class="image-section">
            <h2>🖼️ Image Analysis</h2>
            <div id="imageContainer" class="image-container"></div>
        </div>
    `;

    // Always display stats
    displayVisionStats();

    // Show the full analysis for true parallel results
    displayVisionAnalysis();
}

// Display vision stats
function displayVisionStats() {
    const statsGrid = document.getElementById('statsGrid');
    
    // Check if statsGrid exists (might not exist during visualization)
    if (!statsGrid) {
        console.log('Stats grid not found - likely showing visualization results');
        return;
    }
    
    const analysisResult = window.analysisResult || {};
    const modelUsed = analysisResult.model_used || 'Unknown';
    const processingTime = analysisResult.processing_time_ms || 0;
    const shelves = analysisResult.shelves || [];
    const emptySpaces = analysisResult.empty_spaces || [];
    const scores = analysisResult.analysis_scores || {};

    // Count total products across all shelves
    let totalProducts = 0;
    shelves.forEach(shelf => {
        if (shelf.products) {
            totalProducts += shelf.products.length;
        }
    });
    
    // Check if display was detected
    const isDisplayDetected = analysisResult.is_display_detected !== false;

    statsGrid.innerHTML = `
        <div class="stat-card">
            <div class="stat-number">${modelUsed.toUpperCase()}</div>
            <div>Model Used</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">${Math.round(processingTime)}ms</div>
            <div>Processing Time</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">${totalProducts}</div>
            <div>Products Found</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">${isDisplayDetected ? '✅' : '❌'}</div>
            <div>Display Detected</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">${analysisResult.total_score || scores.overall_score?.value || 'N/A'}</div>
            <div>Total Score</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">${shelves.length}</div>
            <div>Shelf Levels</div>
        </div>
    `;
}

// Display vision analysis
function displayVisionAnalysis() {
    const imageContainer = document.getElementById('imageContainer');
    const analysisResult = window.analysisResult || {};
    
    // Debug logging
    console.log('displayVisionAnalysis called with:', analysisResult);
    console.log('Analysis result keys:', analysisResult ? Object.keys(analysisResult) : 'No result');
    console.log('Empty spaces:', analysisResult?.empty_spaces);
    console.log('Scores:', analysisResult?.scores);
    console.log('Promo matching:', analysisResult?.promo_matching);
    
    // Check if imageContainer exists (might not exist during visualization)
    if (!imageContainer) {
        console.log('Image container not found - likely showing visualization results');
        return;
    }
    
    if (!analysisResult || Object.keys(analysisResult).length === 0) {
        console.warn('No analysis result available');
        return;
    }
    
    // Create image element
    const img = document.createElement('img');
    img.className = 'analyzed-image';
    img.src = URL.createObjectURL(selectedFile);
    
    img.onload = function() {
        // Clear container
        imageContainer.innerHTML = '';
        
        // Add analysis results as text overlay
        const analysisDiv = document.createElement('div');
        analysisDiv.style.cssText = `
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            text-align: left;
        `;
        
        let analysisHTML = `<h3>🤖 Vision Analysis Results</h3>`;
        
        // Check if display was detected
        const isDisplayDetected = analysisResult.is_display_detected !== false;
        if (!isDisplayDetected) {
            analysisHTML += `<div style="color: #d32f2f; font-weight: bold; margin: 10px 0;">❌ No store display detected in this image</div>`;
            analysisDiv.innerHTML = analysisHTML;
            imageContainer.appendChild(img);
            imageContainer.appendChild(analysisDiv);
            return;
        }
        
        // Show banners if detected
        if (analysisResult.banners && analysisResult.banners.length > 0) {
            analysisHTML += `<h4>🏷️ Promotional Banners:</h4>`;
            analysisResult.banners.forEach((banner, index) => {
                analysisHTML += `
                    <div style="margin: 5px 0; padding: 10px; background: #f8f9fa; border-radius: 5px; border-left: 3px solid #007bff;">
                        <div style="font-weight: bold; color: #007bff;">Banner ${index + 1}</div>
                        <div><strong>Text:</strong> ${banner.text}</div>
                        <div><strong>Position:</strong> ${banner.position}</div>
                        <div><strong>Visibility:</strong> ${banner.visibility}/100</div>
                        <div><strong>Confidence:</strong> ${banner.confidence}%</div>
                    </div>
                `;
            });
        } else if (analysisResult.banners && analysisResult.banners.length === 0) {
            analysisHTML += `
                <div style="margin: 5px 0; padding: 10px; background: #f8f9fa; border-radius: 5px; border-left: 3px solid #6c757d;">
                    <h4>🏷️ Promotional Banners:</h4>
                    <div style="color: #6c757d;">No promotional banners detected</div>
                </div>
            `;
        }
        
        // Show shelves and products
        if (analysisResult.shelves && analysisResult.shelves.length > 0) {
            analysisHTML += `<h4>📚 Detected Shelves:</h4>`;
            analysisResult.shelves.forEach((shelf, index) => {
                analysisHTML += `
                    <div style="margin: 10px 0; padding: 10px; background: white; border-radius: 5px;">
                        <strong>Shelf ${shelf.shelf_position || index + 1}</strong> 
                        (Visibility: ${shelf.shelf_visibility || 'N/A'}%)
                        <ul style="margin: 5px 0; padding-left: 20px;">
                `;
                
                // Handle both old format (products) and new format (goods)
                const products = shelf.products || shelf.goods || [];
                if (products.length > 0) {
                    products.forEach(product => {
                        const fullName = product.fullName || product.full_name || product.name;
                        const confidence = product.confidence ? Math.round(product.confidence * 100) : 'N/A';
                        analysisHTML += `<li>${product.name} (${fullName}) - Qty: ${product.count}${confidence !== 'N/A' ? `, Confidence: ${confidence}%` : ''}</li>`;
                    });
                } else {
                    analysisHTML += `<li><em>No products detected</em></li>`;
                }
                
                analysisHTML += `</ul></div>`;
            });
        }
        
        // Show empty spaces
        if (analysisResult.empty_spaces && analysisResult.empty_spaces.length > 0) {
            analysisHTML += `<h4>🔴 Empty Spaces:</h4>`;
            analysisResult.empty_spaces.forEach(space => {
                const areaPercentage = space.area_percentage || space.areaPercentage || 0;
                const confidence = space.confidence || 0;
                analysisHTML += `
                    <div style="margin: 5px 0; padding: 8px; background: #ffebee; border-radius: 5px;">
                        <strong>Shelf ${space.shelf_level || space.shelfLevel || 'Unknown'}</strong> - ${areaPercentage.toFixed(1)}% empty 
                        (Confidence: ${Math.round(confidence * 100)}%)
                    </div>
                `;
            });
        }
        
        // Show scores (handle both old and new format)
        const scores = analysisResult.scores || analysisResult.analysis_scores;
        if (scores) {
            analysisHTML += `<h4>📊 Analysis Scores:</h4>`;
            
            // Show total score
            if (analysisResult.total_score !== undefined) {
                analysisHTML += `
                    <div style="margin: 5px 0; padding: 8px; background: #e8f5e8; border-radius: 5px;">
                        <strong>TOTAL SCORE:</strong> ${analysisResult.total_score}/100
                    </div>
                `;
            }
            
            // Show general comment
            if (analysisResult.general_comment) {
                analysisHTML += `
                    <div style="margin: 5px 0; padding: 8px; background: #fff3e0; border-radius: 5px;">
                        <strong>General Comment:</strong> ${analysisResult.general_comment}
                    </div>
                `;
            }
            
            // Show individual scores (new format)
            const scoreFields = [
                'banner_visibility', 'product_filling', 'promo_match', 'product_neatness',
                'display_cleanliness', 'shelf_arrangement', 'overall_score'
            ];
            
            scoreFields.forEach(field => {
                const score = scores[field];
                if (score && score.value !== undefined) {
                    const scoreValue = score.value || 0;
                    const scoreComment = score.comment || 'No comment available';
                    analysisHTML += `
                        <div style="margin: 5px 0; padding: 8px; background: white; border-radius: 5px;">
                            <strong>${field.replace(/([A-Z])/g, ' $1').toUpperCase()}:</strong> ${scoreValue}/100
                            <br><small style="color: #666;">${scoreComment}</small>
                        </div>
                    `;
                }
            });
            
            // Handle old format scores if present
            if (scores.overall_score && scores.overall_score.value !== undefined) {
                analysisHTML += `
                    <div style="margin: 5px 0; padding: 8px; background: white; border-radius: 5px;">
                        <strong>OVERALL SCORE:</strong> ${scores.overall_score.value}/100
                        <br><small style="color: #666;">${scores.overall_score.comment}</small>
                    </div>
                `;
            }
        }
        

        
        analysisDiv.innerHTML = analysisHTML;
        imageContainer.appendChild(img);
        imageContainer.appendChild(analysisDiv);
    };
}

// Utility functions
function showLoading(message) {
    const loadingSection = document.getElementById('loadingSection');
    loadingSection.style.display = 'block';
    
    // Check if this is a streaming message
    if (message.includes('Streaming')) {
        loadingSection.innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                <p>${message}</p>
                <div id="streamContent" style="margin-top: 10px; font-size: 14px; color: #666;"></div>
            </div>
        `;
    } else {
        loadingSection.innerHTML = `
            <div class="loading">
                <div class="spinner"></div>
                <p>${message}</p>
            </div>
        `;
    }
}

function hideLoading() {
    document.getElementById('loadingSection').style.display = 'none';
}

function hideResults() {
    document.getElementById('resultsSection').style.display = 'none';
}

function showError(message) {
    const loadingSection = document.getElementById('loadingSection');
    loadingSection.innerHTML = `<div class="error">${message}</div>`;
}

function clearPreviousResults() {
    const resultsSection = document.getElementById('resultsSection');
    const existingStreamContent = resultsSection.querySelector('#streamContent');
    if (existingStreamContent) {
        existingStreamContent.remove();
    }
}

function showLoadingState(container) {
    container.innerHTML = `
        <div style="text-align: left; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <h4>🔍 Analyzing Image...</h4>
            <div style="background: #e7f3ff; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff;">
                <strong>Processing...</strong> The AI model is analyzing your shelf image. Results will appear here as they become available.
            </div>
        </div>
    `;
}

function showRawContent(content) {
    document.getElementById('loadingSection').innerHTML = `
        <div style="text-align: left; padding: 20px; background: #f8f9fa; border-radius: 10px;">
            <h4>Analysis Complete:</h4>
            <pre style="white-space: pre-wrap; font-family: inherit;">${content}</pre>
        </div>
    `;
}

// Visualization function
async function visualizeCropping() {
    if (!selectedFile) return;

    showLoading('🎯 Generating visualization...');
    hideResults();

    try {
        const formData = new FormData();
        formData.append('image', selectedFile);

        const response = await fetch('/visualize-cropping', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        // Get headers for additional info
        const detectionMethod = response.headers.get('X-Detection-Method');
        const shelfCount = response.headers.get('X-Shelf-Count');
        const totalObjects = response.headers.get('X-Total-Objects');
        const objectTypes = response.headers.get('X-Object-Types');

        // Convert response to blob
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);

        // Display the visualization
        hideLoading();
        const resultsSection = document.getElementById('resultsSection');
        resultsSection.style.display = 'block';
        
        // Preserve the original structure but add visualization
        resultsSection.innerHTML = `
            <!-- Stats Grid -->
            <div class="stats-section">
                <h2>📊 Detection Statistics</h2>
                <div id="statsGrid" class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-number">${totalObjects}</div>
                        <div>Total Objects</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${shelfCount}</div>
                        <div>Main Frames</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${detectionMethod}</div>
                        <div>Detection Method</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">${objectTypes ? objectTypes.split(',').length : 0}</div>
                        <div>Object Types</div>
                    </div>
                </div>
            </div>

            <!-- Visualization Section -->
            <div class="visualization-section">
                <h2>🎯 YOLO Detection Visualization</h2>
                <div class="visualization-info">
                    <p><strong>Detection Method:</strong> ${detectionMethod}</p>
                    <p><strong>Total Objects Detected:</strong> ${totalObjects}</p>
                    <p><strong>Main Frame Regions:</strong> ${shelfCount}</p>
                    <p><strong>Object Types:</strong> ${objectTypes}</p>
                </div>
                <div class="visualization-image">
                    <img src="${imageUrl}" alt="YOLO detection visualization" style="max-width: 100%; height: auto; border: 2px solid #ddd; border-radius: 8px;">
                </div>
                <div class="visualization-legend">
                    <p><span style="color: #ffffff; font-weight: bold; background: #000; padding: 2px 6px; border-radius: 3px;">White Border</span> = Main Frame (shelf region)</p>
                    <p><span style="color: #ff0000; font-weight: bold;">Red Boxes</span> = Bottles</p>
                    <p><span style="color: #00ff00; font-weight: bold;">Green Boxes</span> = Cans</p>
                    <p><span style="color: #0000ff; font-weight: bold;">Blue Boxes</span> = Packages</p>
                    <p><span style="color: #ffff00; font-weight: bold;">Yellow Boxes</span> = Fruits</p>
                    <p><span style="color: #ff00ff; font-weight: bold;">Magenta Boxes</span> = Boxes</p>
                    <p><span style="color: #808080; font-weight: bold;">Gray Boxes</span> = Other objects</p>
                </div>
                <div class="visualization-actions" style="text-align: center; margin-top: 20px;">
                    <button onclick="analyzeImage()" class="btn btn-primary">
                        🔍 Run Full Analysis
                    </button>
                    <p style="margin-top: 10px; color: #666; font-size: 14px;">
                        Click above to get detailed Gemini analysis of the detected products
                    </p>
                </div>
            </div>
        `;

    } catch (error) {
        hideLoading();
        showError(`Visualization failed: ${error.message}`);
    }
}



 