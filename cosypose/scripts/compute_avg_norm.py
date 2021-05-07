import torch
from cosypose.datasets.datasets_cfg import make_object_dataset
from cosypose.lib3d.rigid_mesh_database import MeshDataBase
from cosypose.lib3d.distances import dists_add, dists_add_symmetric

def compute_errors(TXO_pred, TXO_gt, mesh_db, labels, error_type='ADD-S'):
    '''
    TXO_pred: torch.Size([1, 4, 4])
    TXO_gt: torch.Size([1, 4, 4])
    labels: array(['obj_000001'], dtype=object)
    '''
    TXO_pred = TXO_pred.to('cuda') if not TXO_pred.is_cuda else TXO_pred
    TXO_gt = TXO_gt.to('cuda') if not TXO_gt.is_cuda else TXO_gt
    meshes = mesh_db.select(labels)

    assert len(labels) == 1
    n_points = mesh_db.infos[labels[0]]['n_points']
    points = meshes.points[:, :n_points]

    if error_type.upper() == 'ADD':
        dists = dists_add(TXO_pred, TXO_gt, points)
    elif error_type.upper() == 'ADD-S':
        dists = dists_add_symmetric(TXO_pred, TXO_gt, points)
    elif error_type.upper() == 'ADD(-S)':
        ids_nosym, ids_sym = [], []
        for n, label in enumerate(labels):
            if self.mesh_db.infos[label]['is_symmetric']:
                ids_sym.append(n)
            else:
                ids_nosym.append(n)
        dists = torch.empty((len(TXO_pred), points.shape[1], 3), dtype=TXO_pred.dtype, device=TXO_pred.device)
        if len(ids_nosym) > 0:
            dists[ids_nosym] = dists_add(TXO_pred[ids_nosym], TXO_gt[ids_nosym], points[ids_nosym])
        if len(ids_sym) > 0:
            dists[ids_sym] = dists_add_symmetric(TXO_pred[ids_sym], TXO_gt[ids_sym], points[ids_sym])
    else:
        raise ValueError("Error not supported", error_type)

    errors = dict()
    errors['norm_avg'] = torch.norm(dists, dim=-1, p=2).mean(-1)
    # errors['xyz_avg'] = dists.abs().mean(dim=-2)
    # errors['TCO_xyz'] = (TXO_pred[:, :3, -1] - TXO_gt[:, :3, -1]).abs()
    # errors['TCO_norm'] = torch.norm(TXO_pred[:, :3, -1] - TXO_gt[:, :3, -1], dim=-1, p=2)
    return errors


def main():
    object_ds_name = 'kuartis.eval'
    object_ds = make_object_dataset(object_ds_name)
    mesh_db = MeshDataBase.from_object_ds(object_ds)
    mesh_db = mesh_db.batched().cuda().float()

    labels = ['obj_000001']
    pose1 = torch.tensor([[[ 0.7373,  0.2606, -0.6233,  0.0329],
                            [-0.4774,  0.8538, -0.2077, -0.0669],
                            [ 0.4781,  0.4507,  0.7539,  0.8221],
                            [ 0.0000,  0.0000,  0.0000,  1.0000]]])
    
    pose2 = torch.tensor([[[ 0.7544,  0.2588, -0.6033,  0.0328],
                            [-0.4613,  0.8628, -0.2068, -0.0668],
                            [ 0.4670,  0.4343,  0.7703,  0.8207],
                            [ 0.0000,  0.0000,  0.0000,  1.0000]]])

    errors = compute_errors(pose1, pose2, mesh_db, labels)
    print(errors['norm_avg'].item())


if __name__ == '__main__':
    main()