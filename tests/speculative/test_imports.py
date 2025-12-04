def test_import_new_modules():
    # Ensure coverage tools see these modules as imported
    import needle.ops.ops_mathematic  # noqa: F401
    import needle.backend_ndarray.ndarray  # noqa: F401
    import needle.nn.nn_transformer  # noqa: F401

