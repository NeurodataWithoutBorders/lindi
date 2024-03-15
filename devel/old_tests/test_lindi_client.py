from lindi import LindiClient, LindiGroup, LindiDataset


def test_lindi_client():
    client = LindiClient.from_file("example_0.zarr.json")

    for k, v in client.attrs.items():
        print(f"{k}: {v}")

    for k in client.keys():
        print(k)

    acquisition = client["acquisition"]
    assert isinstance(acquisition, LindiGroup)
    for k in acquisition.keys():
        print(k)

    x = client["acquisition/ElectricalSeriesAp"]["data"]
    assert isinstance(x, LindiDataset)

    print(x.shape)
    print(x[:5])

    general = client["general"]
    assert isinstance(general, LindiGroup)
    for k in general.keys():
        a = general[k]
        if isinstance(a, LindiDataset):
            print(f"{k}: {a.shape}")
            print(a[()])


if __name__ == "__main__":
    test_lindi_client()
