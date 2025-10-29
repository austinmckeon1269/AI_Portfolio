import numpy as np
from src.data_gen import generate_synthetic

def test_gen_shape():
    df = generate_synthetic(1000)
    assert list(df.columns) == ["temp","vib","curr","rpm","failure"]
    assert len(df) == 1000
