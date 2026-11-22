from layers.screening import ScreeningLayer


def test_screening_layer_passes_data_source_configuration():
    layer = ScreeningLayer({"data": {"primary_source": "csv", "max_retries": 0}})
    assert layer.data_fetcher.primary_source == "csv"
    assert layer.data_fetcher.max_retries == 0
