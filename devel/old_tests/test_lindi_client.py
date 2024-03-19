from lindi import LindiZarrWrapper, LindiZarrWrapperGroup, LindiZarrWrapperDataset


def test_lindi_client():
    client = LindiZarrWrapper.from_file("example_0.zarr.json")

    for k, v in client.attrs.items():
        print(f"{k}: {v}")

    for k in client.keys():
        print(k)

    acquisition = client["acquisition"]
    assert isinstance(acquisition, LindiZarrWrapperGroup)
    for k in acquisition.keys():
        print(k)

    aa = client["acquisition/ElectricalSeriesAp"]
    assert isinstance(aa, LindiZarrWrapperGroup)
    x = aa["data"]
    assert isinstance(x, LindiZarrWrapperDataset)

    print(x.shape)
    print(x[:5])

    general = client["general"]
    assert isinstance(general, LindiZarrWrapperGroup)
    for k in general.keys():
        a = general[k]
        if isinstance(a, LindiZarrWrapperDataset):
            print(f"{k}: {a.shape}")
            print(a[()])


if __name__ == "__main__":
    test_lindi_client()
