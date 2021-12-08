import easydict

def get_args():
     args = easydict.EasyDict({
          # general
          "root_dir": "./",
          "exp_name": "1206_wo_mask_loss",

          # Training
          "N_rays": 800,
          "N_src": 3,
          "lr": 0.0005,
          "n_iters": 100000,
          "pe_dim": 10,

          # Testing
          "chunk_size": 800*4,

          # Rendering
          "N_samples": 32,
          "N_importance": 16,
          "inv_uniform": True,
          "det": True,
          "render_stride": 1,
          "ray_sampling_mode": "uniform",    # uniform / center
          "center_ratio": 0.8,

          # Tensorboard
          "log_img": 2000,
          "log_loss": 100,
          "log_weight": 5000,
     })

     return args