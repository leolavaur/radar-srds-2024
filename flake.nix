{
  description = "Trust-FIDS experiments";
  
  inputs = {
    # main nixpkgs
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
 
    # flake-utils
    utils.url = "github:numtide/flake-utils";
  };
  
  outputs = { self, nixpkgs, utils }:

    utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { 
          inherit system;
          config.allowUnfree = true;
        };
      in rec {
        devShell = pkgs.mkShell {
          name = "tensorflow-cuda-shell";

          buildInputs = with pkgs; [ 
            poetry
            python310
          ];

          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.libGL}/lib:${pkgs.libGLU}/lib:${pkgs.freeglut}/lib:${pkgs.xorg.libX11}/lib:${pkgs.cudaPackages.cudatoolkit}/lib:${pkgs.cudaPackages.cudnn}/lib:${pkgs.cudaPackages.cudatoolkit.lib}/lib:${pkgs.cudaPackages.cudatoolkit}/nvvm/libdevice/
            export XLA_FLAGS=--xla_gpu_cuda_data_dir=${pkgs.cudaPackages.cudatoolkit}
            

            # Setup a trap to kill all child processes when this script exits
            # see: https://stackoverflow.com/a/2173421
            trap "trap - SIGTERM && pkill -P $$" SIGINT SIGTERM 

            ${packages.autojupyter}/bin/autojupyter
          '';

        };

        packages."autojupyter" = pkgs.writeShellScriptBin "autojupyter" ''
            # Typically unused port range (https://en.wikipedia.org/wiki/List_of_TCP_and_UDP_port_numbers)
            PORT=3$(id -u)
            
            # check if jupyter is already running
            if [[ -z $(pgrep -f jupyter-notebook -u $(id -u)) ]]; then
              echo "Starting new Jupyter instance..."

              JUPYTER_TOKEN="jupyter" ${pkgs.poetry}/bin/poetry --directory ./exps run jupyter notebook \
                --port=$PORT \
                --NotebookApp.allow_origin='*' \
                --NotebookApp.iopub_msg_rate_limit=3000.0 \
                --no-browser > /dev/null 2>&1 3>&1 &
            else
              echo "Jupyter is already running!"
            fi
            echo "Jupyter instance: http://localhost:$PORT/?token=jupyter"
          '';

        # apps.default = {
        #   type = "app";
        #   program = "${packages.autojupyter}/bin/autojupyter";
        # };

      }
    );

}
