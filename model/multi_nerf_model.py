class SDFGridModel(torch.nn.Module):
    def __init__(self,
                 config,
                 device,
                 bounds,
                 margin=0.1,
                 ):
        super(SDFGridModel, self).__init__()
        self.device = device
        # Simple decoder
        # 计算世界坐标维度、体积原点、体素维度
        world_dims, volume_origin, voxel_dims = compute_world_dims(bounds,
                                                                   config["voxel_sizes"],
                                                                   len(config["voxel_sizes"]),
                                                                   margin=margin,
                                                                   device=device)

        self.world_dims = world_dims
        self.volume_origin = volume_origin
        self.voxel_dims = voxel_dims

        grid_dim = (torch.tensor(config["sdf_feature_dim"]) + torch.tensor(config["rgb_feature_dim"])).tolist()
        self.grid = MultiGrid(voxel_dims, grid_dim).to(device)

        self.sigle_mlp = config['single_mlp']
        self.nerf_mlp = NerfMLP(config['decoder']['NerfMLP'])
        self.prop_mlp = nerf_mlp if self.single_mlp else PropMLP(config['decoder']['PropMLP'])
        self.config = config
        self.num_levels = config['num_levels']
        self.num_prop_samples = config['num_prop_samples']
        self.num_nerf_samples = config['num_nerf_samples']

    def forward(self, rays_o, rays_d, target_rgb_select, target_depth_select, inv_s=None, smoothness_std=0., iter=0):
        for i_level in range(self.num_levels):
            is_prop = i_level<(self.num_levels - 1)
            num_samples = self.num_prop_samples if is_prop else self.num_nerf_samples
            # 将标准化距离转换为度量距离。
            tdist = s_to_t(sdist)
            gaussians = cast_rays(
                tdist,
                rays.origins,
                rays.directions,
                rays.radii,
                self.ray_shape,
                diag=False)
            mlp = prop_mlp if is_prop else nerf_mlp
            key, rng = random_split(rng)
            ray_results = mlp(
                key,
                gaussians,
                viewdirs=rays.viewdirs if self.use_viewdirs else None,
                imageplane=rays.imageplane,
                glo_vec=None if is_prop else glo_vec,
                exposure=rays.exposure_values,
            )
            # 获取用于体积渲染（以及其他损失函数）的权重。
            weights = compute_alpha_weights(
                ray_results['density'],
                tdist,
                rays.directions,
                opaque_background=self.opaque_background,
            )[0]

        ret = {
            "rgb_loss": rgb_loss,
            "depth_loss": depth_loss,
            "sdf_loss": rend_dict["sdf_loss"],
            "fs_loss": rend_dict["fs_loss"],
            "normal_regularisation_loss": rend_dict["normal_regularisation_loss"],
            "normal_supervision_loss": rend_dict["normal_supervision_loss"],
            "eikonal_loss": rend_dict["eikonal_loss"],
            "psnr": psnr
        }

        return ret


def cast_rays(tdist, origins, directions):
    """
    Cast rays and compute sample points.

    Args:
        tdist: float array, the "fencepost" distances along the ray.
        origins: float array, the ray origin coordinates.
        directions: float array, the ray direction vectors.

    Returns:
        An array of sample points along the rays.
    """
    # Get the starting and ending distances for each segment
    t0 = tdist[..., :-1]
    t1 = tdist[..., 1:]

    # Compute the midpoint of each segment
    midpoints = (t0 + t1) / 2.0

    # Calculate sample points
    sample_points = origins[..., None, :] + directions[..., None, :] * midpoints[..., None]

    return sample_points

def compute_alpha_weights(density, tdist, dirs, opaque_background=False):
  """Helper function for computing alpha compositing weights."""
  t_delta = tdist[..., 1:] - tdist[..., :-1]
  delta = t_delta * jnp.linalg.norm(dirs[..., None, :], axis=-1)
  density_delta = density * delta

  if opaque_background:
    # Equivalent to making the final t-interval infinitely wide.
    density_delta = jnp.concatenate([
        density_delta[..., :-1],
        jnp.full_like(density_delta[..., -1:], jnp.inf)
    ],
                                    axis=-1)

  alpha = 1 - jnp.exp(-density_delta)
  trans = jnp.exp(-jnp.concatenate([
      jnp.zeros_like(density_delta[..., :1]),
      jnp.cumsum(density_delta[..., :-1], axis=-1)
  ],
                                   axis=-1))
  weights = alpha * trans
  return weights, alpha, trans