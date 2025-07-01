#!/usr/bin/env python3

import torch
from llava.train.custom_callbacks import VideoDescriptionCallback
from llava.utils import rank0_print

def test_video_description_callback():
    """Test the video description callback with mock data"""
    
    # Create mock inputs that simulate what the training loop provides
    mock_inputs = {
        'prompts': [
            "This video shows a person walking through a retail store, looking at various items on the shelves. The person appears to be shopping normally, examining products and occasionally picking up items to look at them more closely.",
            "A surveillance video of a customer in a department store. The individual is browsing through clothing racks, trying on different items, and interacting with store staff. The video captures typical shopping behavior."
        ],
        'images': [torch.randn(16, 3, 224, 224), torch.randn(16, 3, 224, 224)],  # Mock video frames
        'image_sizes': [(224, 224), (224, 224)],
        'modalities': ['video', 'video']
    }
    
    # Create the callback
    callback = VideoDescriptionCallback(log_every_n_steps=1)
    
    # Mock training state
    class MockState:
        def __init__(self):
            self.global_step = 30
    
    class MockArgs:
        def __init__(self):
            pass
    
    # Test the callback
    print("Testing VideoDescriptionCallback...")
    callback.on_step_end(
        args=MockArgs(),
        state=MockState(),
        control={},
        model=None,  # We don't need the actual model for this test
        inputs=mock_inputs
    )
    
    print("✅ Video description callback test completed!")

if __name__ == "__main__":
    test_video_description_callback() 