from fastapi import APIRouter
from typing import List
from ..schemas import DatasetMetadata, ClassDistribution, DatasetItem

router = APIRouter(prefix="/datasets", tags=["datasets"])

@router.get("", response_model=List[DatasetMetadata])
def get_datasets_metadata():
    """
    Returns structural details, sizes, class distributions, and sample data records for research datasets.
    """
    return [
        # --- Text Datasets ---
        DatasetMetadata(
            name="IMDB Reviews",
            category="text",
            size=50000,
            description="Highly popular binary classification dataset of movie reviews from IMDB, heavily utilized for evaluating word embeddings.",
            class_distribution=[
                ClassDistribution(label="Positive", count=25000),
                ClassDistribution(label="Negative", count=25000),
                ClassDistribution(label="Neutral", count=0),
            ],
            sample_records=[
                DatasetItem(id=1, text_content="An absolute masterpiece. The cinematography and acting are top notch.", true_label="Positive"),
                DatasetItem(id=2, text_content="Boring, slow-paced, and predictable. Waste of two hours.", true_label="Negative"),
                DatasetItem(id=3, text_content="It had great potential, but falls flat in the final act.", true_label="Negative")
            ]
        ),
        DatasetMetadata(
            name="Amazon Reviews",
            category="text",
            size=142000,
            description="Multi-domain product feedback logs. Used for evaluating cross-domain adaptability of sentiment classifiers.",
            class_distribution=[
                ClassDistribution(label="Positive", count=78000),
                ClassDistribution(label="Negative", count=34000),
                ClassDistribution(label="Neutral", count=30000),
            ],
            sample_records=[
                DatasetItem(id=1, text_content="The product works exactly as advertised. Very sturdy design.", true_label="Positive"),
                DatasetItem(id=2, text_content="Terrible build quality. Broke within three days of normal use.", true_label="Negative"),
                DatasetItem(id=3, text_content="Decent product but the shipping was extremely delayed.", true_label="Neutral")
            ]
        ),
        DatasetMetadata(
            name="Sentiment140",
            category="text",
            size=1600000,
            description="Large-scale database of tweets. Utilizes emoticons as distant supervision labels to check performance on informal grammar.",
            class_distribution=[
                ClassDistribution(label="Positive", count=800000),
                ClassDistribution(label="Negative", count=800000),
                ClassDistribution(label="Neutral", count=0),
            ],
            sample_records=[
                DatasetItem(id=1, text_content="Beautiful morning today! Can't wait to start the weekend.", true_label="Positive"),
                DatasetItem(id=2, text_content="Ugh, stuck in terrible traffic. Going to be late for the presentation.", true_label="Negative"),
                DatasetItem(id=3, text_content="Heading home from work now.", true_label="Neutral")
            ]
        ),
        DatasetMetadata(
            name="US Airlines Twitter Sentiment",
            category="text",
            size=14485,
            description="Twitter reviews regarding airline services. Evaluates ability to isolate customer complaints and expressions of anger.",
            class_distribution=[
                ClassDistribution(label="Positive", count=2363),
                ClassDistribution(label="Negative", count=9178),
                ClassDistribution(label="Neutral", count=2944),
            ],
            sample_records=[
                DatasetItem(id=1, text_content="Thank you for resolving my ticket so quickly. Great service!", true_label="Positive"),
                DatasetItem(id=2, text_content="Flight cancelled and no one is at the counter. Extremely poor communication.", true_label="Negative"),
                DatasetItem(id=3, text_content="Are there any direct flights from JFK to LAX tomorrow?", true_label="Neutral")
            ]
        ),
        
        # --- Multimodal Datasets ---
        DatasetMetadata(
            name="CMU-MOSI",
            category="multimodal",
            size=2199,
            description="CMU Multimodal Opinion Sentiment Intensity dataset. Contains video clips annotated with aligned text transcripts, facial features, and audio tones.",
            class_distribution=[
                ClassDistribution(label="Positive", count=1120),
                ClassDistribution(label="Negative", count=879),
                ClassDistribution(label="Neutral", count=200),
            ],
            sample_records=[
                DatasetItem(id=1, text_content="I really enjoyed this movie, it had great pacing.", media_url="cmu_mosi_001.mp4", true_label="Positive"),
                DatasetItem(id=2, text_content="The screen is way too small and I feel annoyed.", media_url="cmu_mosi_002.mp4", true_label="Negative"),
                DatasetItem(id=3, text_content="Well, we checked out the place yesterday afternoon.", media_url="cmu_mosi_003.mp4", true_label="Neutral")
            ]
        ),
        DatasetMetadata(
            name="MuSe",
            category="multimodal",
            size=4500,
            description="Multimodal Sentiment Analysis in the Wild. Measures sentiment in longer conversational context and real-world noise environments.",
            class_distribution=[
                ClassDistribution(label="Positive", count=1850),
                ClassDistribution(label="Negative", count=1750),
                ClassDistribution(label="Neutral", count=900),
            ],
            sample_records=[
                DatasetItem(id=1, text_content="This is amazing! The layout is perfect.", media_url="muse_clip_09.mp4", true_label="Positive"),
                DatasetItem(id=2, text_content="The volume dial is stuck and the noise is unbearable.", media_url="muse_clip_14.mp4", true_label="Negative"),
                DatasetItem(id=3, text_content="We are testing the device under room temperature settings.", media_url="muse_clip_32.mp4", true_label="Neutral")
            ]
        ),
        DatasetMetadata(
            name="GeoCoV19",
            category="multimodal",
            size=20000,
            description="Multimodal social posts relating to COVID-19. Focuses on crisis sentiment monitoring through text and image sharing.",
            class_distribution=[
                ClassDistribution(label="Positive", count=4500),
                ClassDistribution(label="Negative", count=11500),
                ClassDistribution(label="Neutral", count=4000),
            ],
            sample_records=[
                DatasetItem(id=1, text_content="Community food drive was a huge success today!", media_url="geocov_img_01.jpg", true_label="Positive"),
                DatasetItem(id=2, text_content="Lockdowns extended again. Feeling exhausted and isolated.", media_url="geocov_img_02.jpg", true_label="Negative"),
                DatasetItem(id=3, text_content="Local vaccination clinics are updating their operational hours.", media_url="geocov_img_03.jpg", true_label="Neutral")
            ]
        )
    ]
