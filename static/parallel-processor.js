/**
 * Parallel Processing Module using Web Workers
 * This provides true parallelism for shelf analysis
 */

class ParallelProcessor {
    constructor() {
        this.workers = [];
    }

    /**
     * Get shelf coordinates from debug endpoint and process in parallel
     */
    async processImageInParallel(imageFile, model) {
        console.log('🚀 Starting true parallel processing...');
        const startTime = performance.now();

        try {
            // Step 1: Get shelf coordinates from debug endpoint
            console.log('📍 Step 1: Getting shelf coordinates...');
            const shelfCoordinates = await this.getShelfCoordinates(imageFile);
            
            if (!shelfCoordinates || shelfCoordinates.length === 0) {
                throw new Error('No shelves detected');
            }

            console.log(`✅ Found ${shelfCoordinates.length} shelves:`, shelfCoordinates);

            // Step 2: Crop image in JavaScript
            console.log('✂️ Step 2: Cropping image in JavaScript...');
            const croppedImages = await this.cropImageInJavaScript(imageFile, shelfCoordinates);
            
            console.log(`✅ Cropped ${croppedImages.length} shelf images`);

            // Step 3: Process each shelf in parallel using Web Workers
            console.log('⚡ Step 3: Processing shelves in parallel...');
            const results = await this.processShelvesInParallel(croppedImages, model);

            const totalTime = performance.now() - startTime;
            console.log(`🎉 Parallel processing completed in ${totalTime.toFixed(2)}ms`);

            return {
                results: results,
                totalTime: totalTime,
                shelfCount: shelfCoordinates.length
            };

        } catch (error) {
            console.error('❌ Parallel processing failed:', error);
            throw error;
        }
    }

    /**
     * Get shelf coordinates from debug endpoint
     */
    async getShelfCoordinates(imageFile) {
        const formData = new FormData();
        formData.append('image', imageFile);

        const response = await fetch('/debug-shelf-detection', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Debug endpoint failed: ${response.status}`);
        }

        const debugData = await response.json();
        
        if (!debugData.shelf_detection || !debugData.shelf_detection.shelves) {
            throw new Error('No shelf data in debug response');
        }

        return debugData.shelf_detection.shelves.map(shelf => ({
            shelf_id: shelf.shelf_id,
            bounding_box: shelf.bounding_box,
            confidence: shelf.confidence
        }));
    }

    /**
     * Crop image in JavaScript using Canvas
     */
    async cropImageInJavaScript(imageFile, shelfCoordinates) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                
                const croppedImages = [];

                shelfCoordinates.forEach(shelf => {
                    const bbox = shelf.bounding_box;
                    
                    // Set canvas size to shelf dimensions
                    canvas.width = bbox.width;
                    canvas.height = bbox.height;
                    
                    // Clear canvas
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    
                    // Draw cropped section
                    ctx.drawImage(
                        img,
                        bbox.x1, bbox.y1, bbox.width, bbox.height,  // Source rectangle
                        0, 0, bbox.width, bbox.height               // Destination rectangle
                    );
                    
                    // Convert to blob
                    canvas.toBlob((blob) => {
                        croppedImages.push({
                            shelf_id: shelf.shelf_id,
                            image: blob,
                            bounding_box: bbox
                        });
                        
                        // Resolve when all crops are done
                        if (croppedImages.length === shelfCoordinates.length) {
                            resolve(croppedImages);
                        }
                    }, 'image/jpeg', 0.9);
                });
            };
            
            img.onerror = () => reject(new Error('Failed to load image for cropping'));
            img.src = URL.createObjectURL(imageFile);
        });
    }

    /**
     * Process shelves in parallel using Web Workers
     */
    async processShelvesInParallel(croppedImages, model) {
        const workerPromises = croppedImages.map(croppedImage => 
            this.processShelfWithWorker(croppedImage, model)
        );

        console.log(`🚀 Starting ${croppedImages.length} parallel workers...`);
        
        // Wait for all workers to complete
        const results = await Promise.allSettled(workerPromises);
        
        // Process results
        const successfulResults = [];
        const failedResults = [];
        
        results.forEach((result, index) => {
            if (result.status === 'fulfilled') {
                successfulResults.push({
                    shelf_id: croppedImages[index].shelf_id,
                    result: result.value,
                    success: true
                });
            } else {
                failedResults.push({
                    shelf_id: croppedImages[index].shelf_id,
                    error: result.reason,
                    success: false
                });
            }
        });

        console.log(`✅ Completed: ${successfulResults.length} successful, ${failedResults.length} failed`);
        
        return {
            successful: successfulResults,
            failed: failedResults,
            total: croppedImages.length
        };
    }

    /**
     * Process a single shelf using a Web Worker
     */
    async processShelfWithWorker(croppedImage, model) {
        return new Promise((resolve, reject) => {
            // Get the current origin for the worker
            const origin = window.location.origin;
            
            const worker = new Worker(URL.createObjectURL(new Blob([`
                self.onmessage = async function(e) {
                    const { shelfId, imageBlob, model, origin } = e.data;
                    
                    try {
                        const formData = new FormData();
                        formData.append('image', imageBlob);
                        formData.append('model', model);
                        
                        const startTime = performance.now();
                        
                        const response = await fetch(\`\${origin}/analyze-shelf-crop\`, {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (!response.ok) {
                            throw new Error(\`HTTP \${response.status}: \${response.statusText}\`);
                        }
                        
                        const result = await response.json();
                        const processingTime = performance.now() - startTime;
                        
                        self.postMessage({
                            type: 'success',
                            shelfId: shelfId,
                            result: result,
                            processingTime: processingTime
                        });
                        
                    } catch (error) {
                        self.postMessage({
                            type: 'error',
                            shelfId: shelfId,
                            error: error.message
                        });
                    }
                };
            `], { type: 'application/javascript' })));

            worker.onmessage = (e) => {
                const { type, shelfId, result, processingTime, error } = e.data;
                
                if (type === 'success') {
                    console.log(`✅ Shelf ${shelfId} completed in ${processingTime.toFixed(2)}ms`);
                    resolve(result);
                } else {
                    console.error(`❌ Shelf ${shelfId} failed:`, error);
                    reject(new Error(error));
                }
                
                worker.terminate();
            };

            worker.onerror = (error) => {
                console.error(`❌ Worker error for shelf ${croppedImage.shelf_id}:`, error);
                reject(error);
                worker.terminate();
            };

            // Send data to worker
            worker.postMessage({
                shelfId: croppedImage.shelf_id,
                imageBlob: croppedImage.image,
                model: model,
                origin: origin
            });
        });
    }

    /**
     * Combine shelf results into final analysis
     */
    combineShelfResults(shelfResults) {
        const allShelves = [];
        const allEmptySpaces = [];
        let totalProcessingTime = 0;

        shelfResults.successful.forEach(shelfData => {
            const result = shelfData.result;
            
            console.log(`📦 Processing shelf ${shelfData.shelf_id} result:`, result);
            
            // Each shelf analysis should return a single shelf result
            // We need to extract the main shelf data and adjust the position
            if (result.shelves && result.shelves.length > 0) {
                // Take the first (and should be only) shelf from each analysis
                const shelf = result.shelves[0];
                shelf.shelf_position = shelfData.shelf_id;
                allShelves.push(shelf);
                console.log(`✅ Added shelf ${shelfData.shelf_id} with ${shelf.products?.length || 0} products`);
            } else {
                console.warn(`⚠️ Shelf ${shelfData.shelf_id} has no shelves array`);
            }
            
            // Collect empty spaces and adjust shelf levels
            if (result.empty_spaces) {
                result.empty_spaces.forEach(space => {
                    space.shelf_level = shelfData.shelf_id;
                    allEmptySpaces.push(space);
                });
            }
            
            // Accumulate processing time
            totalProcessingTime += shelfData.processingTime || 0;
        });

        // Create combined result
        const combinedResult = {
            shelves: allShelves,
            empty_spaces: allEmptySpaces,
            analysis_scores: this.calculateOverallScores(allShelves, allEmptySpaces),
            processing_time_ms: totalProcessingTime
        };

        return combinedResult;
    }

    /**
     * Calculate overall scores from shelf results
     */
    calculateOverallScores(shelves, emptySpaces) {
        if (shelves.length === 0) {
            return {
                overall_score: 0,
                shelf_visibility: 0,
                product_diversity: 0,
                space_utilization: 0,
                details: {
                    shelf_visibility: { score: 0, reasoning: "No shelves analyzed" },
                    product_diversity: { score: 0, reasoning: "No products found" },
                    space_utilization: { score: 0, reasoning: "No space data available" }
                }
            };
        }

        // Calculate average scores from the actual shelf data
        const avgVisibility = shelves.reduce((sum, shelf) => sum + (shelf.shelf_visibility || 0), 0) / shelves.length;
        
        // Calculate product diversity based on unique products across all shelves
        const allProducts = shelves.flatMap(shelf => shelf.products || []);
        const uniqueProducts = new Set(allProducts.map(p => p.name));
        const productDiversity = Math.min(100, (uniqueProducts.size / Math.max(1, allProducts.length)) * 100);
        
        // Calculate space utilization based on empty spaces
        const totalEmptyPercentage = emptySpaces.reduce((sum, space) => sum + (space.area_percentage || 0), 0);
        const avgEmptyPercentage = emptySpaces.length > 0 ? totalEmptyPercentage / emptySpaces.length : 0;
        const spaceUtilization = Math.max(0, 100 - avgEmptyPercentage);

        const overallScore = Math.round((avgVisibility + productDiversity + spaceUtilization) / 3);

        return {
            overall_score: overallScore,
            shelf_visibility: Math.round(avgVisibility),
            product_diversity: Math.round(productDiversity),
            space_utilization: Math.round(spaceUtilization),
            details: {
                shelf_visibility: { score: Math.round(avgVisibility), reasoning: `Average visibility across ${shelves.length} shelves` },
                product_diversity: { score: Math.round(productDiversity), reasoning: `${uniqueProducts.size} unique products across ${shelves.length} shelves` },
                space_utilization: { score: Math.round(spaceUtilization), reasoning: `${Math.round(avgEmptyPercentage)}% average empty space` }
            }
        };
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ParallelProcessor;
} else {
    window.ParallelProcessor = ParallelProcessor;
} 