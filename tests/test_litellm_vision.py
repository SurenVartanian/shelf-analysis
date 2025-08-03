import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import base64
from PIL import Image
import io

from src.shelf_analyzer.services.litellm_vision_service import (
    LiteLLMVisionService,
    VisionAnalysisConfig,
    VisionModel,
    VisionAnalysisResult
)


class TestLiteLLMVisionService:
    """Test LiteLLM Vision Service"""
    
    @pytest.fixture
    def vision_service(self):
        """Create a vision service instance"""
        config = VisionAnalysisConfig(model=VisionModel.GPT_4O)
        return LiteLLMVisionService(config)
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing"""
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='white')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        return img_bytes.getvalue()
    
    @pytest.mark.asyncio
    async def test_initialization(self, vision_service):
        """Test service initialization"""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            await vision_service.initialize()
            assert vision_service.is_ready() is True
    
    @pytest.mark.asyncio
    async def test_initialization_no_api_key(self, vision_service):
        """Test initialization without API key"""
        with patch.dict('os.environ', {}, clear=True):
            await vision_service.initialize()
            # Service should still initialize but warn about missing API keys
            assert vision_service.is_ready() is True
    
    @pytest.mark.asyncio
    async def test_analyze_shelf_not_ready(self, vision_service, sample_image):
        """Test analysis when service is not ready"""
        with pytest.raises(RuntimeError, match="Service not initialized"):
            await vision_service.analyze_shelf(sample_image)
    
    @pytest.mark.asyncio
    async def test_analyze_shelf_success(self, vision_service, sample_image):
        """Test successful shelf analysis"""
        # Mock the litellm response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '''
        {
          "model_used": "gpt-4o",
          "processing_time_ms": 1234.56,
          "is_display_detected": true,
          "banners": [],
          "shelves": [
            {
              "shelf_position": 1,
              "shelf_visibility": 90,
              "products": [
                {
                  "name": "Test Product",
                  "full_name": "Test Product Brand",
                  "count": 5
                }
              ]
            }
          ],
          "scores": {
            "banner_visibility": {"value": 0, "comment": "No banners detected"},
            "product_filling": {"value": 75, "comment": "Good density"},
            "promo_match": {"value": 60, "comment": "Limited promotional material"},
            "product_neatness": {"value": 80, "comment": "Well organized"},
            "display_cleanliness": {"value": 85, "comment": "Clean display"},
            "shelf_arrangement": {"value": 85, "comment": "Logical arrangement"},
            "overall_score": {"value": 75, "comment": "Overall good"}
          },
          "total_score": 75,
          "general_comment": "Test analysis completed successfully"
        }
        '''
        
        with patch('litellm.acompletion', new_callable=AsyncMock) as mock_litellm:
            mock_litellm.return_value = mock_response
            
            # Initialize service
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
                await vision_service.initialize()
            
            # Test analysis
            result = await vision_service.analyze_shelf(sample_image)
            
            assert result.model_used == VisionModel.GPT_4O.value
            assert len(result.shelves) == 1
            assert result.shelves[0].shelf_position == 1
            assert len(result.shelves[0].products) == 1
            assert result.shelves[0].products[0].name == "Test Product"
            assert result.scores.overall_score.value == 75
    
    @pytest.mark.asyncio
    async def test_analyze_shelf_fallback(self, vision_service, sample_image):
        """Test that invalid JSON raises an exception"""
        # Mock the litellm response with non-JSON content
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is a text response without JSON"
        
        with patch('litellm.acompletion', new_callable=AsyncMock) as mock_litellm:
            mock_litellm.return_value = mock_response
            
            # Initialize service
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
                await vision_service.initialize()
            
            # Test analysis - should raise exception for invalid JSON
            with pytest.raises(Exception):
                await vision_service.analyze_shelf(sample_image)
    
    @pytest.mark.asyncio
    async def test_stream_analysis(self, vision_service, sample_image):
        """Test streaming analysis"""
        # Mock streaming response
        mock_chunks = [
            Mock(choices=[Mock(delta=Mock(content="Analyzing"))]),
            Mock(choices=[Mock(delta=Mock(content=" shelf"))]),
            Mock(choices=[Mock(delta=Mock(content=" image"))])
        ]
        
        async def mock_stream():
            for chunk in mock_chunks:
                yield chunk
        
        with patch('litellm.acompletion') as mock_litellm:
            mock_litellm.return_value = mock_stream()
            
            # Initialize service
            with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
                await vision_service.initialize()
            
            # Test streaming
            chunks = []
            async for chunk in vision_service.analyze_shelf_stream(sample_image):
                chunks.append(chunk)
            
            assert len(chunks) == 3
            assert chunks[0] == "Analyzing"
            assert chunks[1] == " shelf"
            assert chunks[2] == " image"
    
    def test_vision_model_enum(self):
        """Test vision model enum values"""
        assert VisionModel.GPT_4O.value == "gpt-4o"
        assert VisionModel.GEMINI_FLASH.value == "gemini-flash"
        assert VisionModel.GEMINI_FLASH_LITE.value == "gemini-flash-lite"


if __name__ == "__main__":
    pytest.main([__file__]) 