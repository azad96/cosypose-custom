BOP_CONFIG = dict()

ds_key = 'kuatless'
BOP_CONFIG[ds_key] = dict(
    input_resize=(1080, 810),
    render_size=(360, 480),
    urdf_ds_name='{}.cad'.format(ds_key),
    obj_ds_name='{}.cad'.format(ds_key),
    train_pbr_ds_name=['{}.train_pbr_n_20_25_50'.format(ds_key)],
    test_pbr_ds_name=['{}.test_pbr_1080_810'.format(ds_key)],
)

ds_key = 'bm'
BOP_CONFIG[ds_key] = dict(
    input_resize=(1080, 810),
    render_size=(360, 480),
    urdf_ds_name='{}.cad'.format(ds_key),
    obj_ds_name='{}.cad'.format(ds_key),
    train_pbr_ds_name=['{}.train_pbr'.format(ds_key)],
    test_pbr_ds_name=['{}.test_pbr_1080_810'.format(ds_key)],
)

ds_key = 'bm2'
BOP_CONFIG[ds_key] = dict(
    input_resize=(1080, 810),
    render_size=(360, 480),
    urdf_ds_name='{}.cad'.format(ds_key),
    obj_ds_name='{}.cad'.format(ds_key),
    train_pbr_ds_name=['{}.train_pbr_dark'.format(ds_key)],
    test_pbr_ds_name=['{}.test_pbr_1080_810'.format(ds_key)],
    # train_pbr_ds_name=['{}.train_pbr_mix'.format(ds_key)],
    # test_pbr_ds_name=['{}.test4_pbr_1080_810'.format(ds_key)],
)
