from src.data.generator import generate_dataset
from src.data.loader import split_features_target
from src.models.pipeline import create_clinical_pipeline


def test_end_to_end_components_smoke():
    df = generate_dataset(n_samples=20)
    X, y = split_features_target(df)
    pipeline = create_clinical_pipeline()

    assert X.shape == (20, 8)
    assert y.isin([0, 1]).all()
    assert pipeline.steps[-1][0] == "classifier"
