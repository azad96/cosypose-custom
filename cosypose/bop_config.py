BOP_CONFIG = dict()

ds_key = 'kuatless'
BOP_CONFIG[ds_key] = dict(
    input_resize=(1080, 810),
    render_size=(360, 480),
    urdf_ds_name='{}.cad'.format(ds_key),
    obj_ds_name='{}.cad'.format(ds_key),
    train_pbr_ds_name=['{}.train_pbr_noise'.format(ds_key)],
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
    # train_pbr_ds_name=['{}.train_pbr_dark'.format(ds_key)],
    # test_pbr_ds_name=['{}.test_pbr_1080_810'.format(ds_key)],
    train_pbr_ds_name=['{}.train_pbr_mix'.format(ds_key)],
    test_pbr_ds_name=['{}.test4_pbr_1080_810'.format(ds_key)],
)

PBR_DETECTORS = dict(
    hb='detector-bop-hb-pbr--497808',
    icbin='detector-bop-icbin-pbr--947409',
    itodd='detector-bop-itodd-pbr--509908',
    lmo='detector-bop-lmo-pbr--517542',
    tless='detector-bop-tless-pbr--873074',
    tudl='detector-bop-tudl-pbr--728047',
    ycbv='detector-bop-ycbv-pbr--970850',
)

PBR_COARSE = dict(
    hb='coarse-bop-hb-pbr--70752',
    icbin='coarse-bop-icbin-pbr--915044',
    itodd='coarse-bop-itodd-pbr--681884',
    lmo='coarse-bop-lmo-pbr--707448',
    tless='coarse-bop-tless-pbr--506801',
    tudl='coarse-bop-tudl-pbr--373484',
    ycbv='coarse-bop-ycbv-pbr--724183',
)

PBR_REFINER = dict(
    hb='refiner-bop-hb-pbr--247731',
    icbin='refiner-bop-icbin-pbr--841882',
    itodd='refiner-bop-itodd-pbr--834427',
    lmo='refiner-bop-lmo-pbr--325214',
    tless='refiner-bop-tless-pbr--233420',
    tudl='refiner-bop-tudl-pbr--487212',
    ycbv='refiner-bop-ycbv-pbr--604090',
)

SYNT_REAL_DETECTORS = dict(
    tudl='detector-bop-tudl-synt+real--298779',
    tless='detector-bop-tless-synt+real--452847',
    ycbv='detector-bop-ycbv-synt+real--292971',
)

SYNT_REAL_COARSE = dict(
    tudl='coarse-bop-tudl-synt+real--610074',
    tless='coarse-bop-tless-synt+real--160982',
    ycbv='coarse-bop-ycbv-synt+real--822463',
)

SYNT_REAL_REFINER = dict(
    tudl='refiner-bop-tudl-synt+real--423239',
    tless='refiner-bop-tless-synt+real--881314',
    ycbv='refiner-bop-ycbv-synt+real--631598',
)


for k, v in PBR_COARSE.items():
    if k not in SYNT_REAL_COARSE:
        SYNT_REAL_COARSE[k] = v

for k, v in PBR_REFINER.items():
    if k not in SYNT_REAL_REFINER:
        SYNT_REAL_REFINER[k] = v

for k, v in PBR_DETECTORS.items():
    if k not in SYNT_REAL_DETECTORS:
        SYNT_REAL_DETECTORS[k] = v


PBR_INFERENCE_ID = 'bop-pbr--223026'
SYNT_REAL_INFERENCE_ID = 'bop-synt+real--815712'
SYNT_REAL_ICP_INFERENCE_ID = 'bop-synt+real-icp--121351'
SYNT_REAL_4VIEWS_INFERENCE_ID = 'bop-synt+real-nviews=4--419066'
SYNT_REAL_8VIEWS_INFERENCE_ID = 'bop-synt+real-nviews=8--763684'
