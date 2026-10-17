{
  description = "vegaml — typed decisions from a frozen LLM + physics engine (CUDA 12.8 / cu128 build for hq's RTX 3090, driver 570)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, uv2nix, pyproject-nix, pyproject-build-systems, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true; # nvidia cu12 wheels
      };
      lib = nixpkgs.lib;
      python = pkgs.python312;

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

      # Prefer prebuilt wheels — never build torch/cuda from source.
      overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };

      # The manylinux wheels (torch + nvidia-*-cu12 + triton) ship ELF binaries whose
      # RPATHs must be patched to the nix C/C++ runtime. libcuda is the *driver* — it is
      # NOT in the closure (it lives at /run/opengl-driver/lib on the host), so we tell
      # autopatchelf to ignore it and we inject it at runtime in the serve wrapper.
      pyprojectOverrides = final: prev:
        let
          nvidiaNames = builtins.filter
            (n: lib.hasPrefix "nvidia-" n && prev ? ${n})
            (builtins.attrNames prev);
          baseLibs = with pkgs; [ stdenv.cc.cc.lib zlib libGL ];
          # The cu12 wheels dlopen each other (cusolver->cublas/cusparse/nvJitLink) and the
          # host driver (libcuda). We do NOT wire siblings as buildInputs — that poisons
          # pyproject.nix's venv resolver (infinite recursion). Instead ignore all missing
          # at patch time: the wheels are co-located in the final venv's site-packages/nvidia,
          # torch's _load_global_deps preloads them RTLD_GLOBAL in dependency order, and the
          # serve wrapper puts both the venv nvidia dirs and /run/opengl-driver on the path.
          patch = pkg: pkg.overrideAttrs (old: {
            nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ [ pkgs.autoPatchelfHook ];
            buildInputs = (old.buildInputs or [ ]) ++ baseLibs;
            autoPatchelfIgnoreMissingDeps = true;
          });
          nvidiaOverrides = lib.genAttrs nvidiaNames (n: patch prev.${n});
        in
        nvidiaOverrides // {
          torch = patch prev.torch;
          torchvision = patch prev.torchvision;
          triton = patch prev.triton;
        };

      pythonSet = (pkgs.callPackage pyproject-nix.build.packages { inherit python; }).overrideScope
        (lib.composeManyExtensions [
          pyproject-build-systems.overlays.default
          overlay
          pyprojectOverrides
        ]);

      # Include the `serve` extra (fastapi + uvicorn) so vegaml-serve runs.
      venv = pythonSet.mkVirtualEnv "vegaml-env"
        (workspace.deps.default // { vegaml = [ "serve" ]; });

      # Runtime wrapper: expose the host NVIDIA driver (libcuda) that the wheels dlopen.
      # /run/opengl-driver/lib = host NVIDIA driver (libcuda) on NixOS; the nvidia/*/lib
      # dirs = the cu12 runtime wheels co-located in the venv.
      vegaml-serve = pkgs.writeShellScriptBin "vegaml-serve" ''
        nvlibs=$(echo ${venv}/lib/python*/site-packages/nvidia/*/lib | tr ' ' ':')
        export LD_LIBRARY_PATH="/run/opengl-driver/lib:$nvlibs''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        exec ${venv}/bin/vegaml-serve "$@"
      '';
    in
    {
      packages.${system} = {
        default = venv;
        inherit venv vegaml-serve;
      };

      apps.${system}.default = {
        type = "app";
        program = "${vegaml-serve}/bin/vegaml-serve";
      };

      devShells.${system}.default = pkgs.mkShell {
        packages = [ venv pkgs.uv ];
        shellHook = ''
          export LD_LIBRARY_PATH="/run/opengl-driver/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        '';
      };

      # NixOS module: services.vegaml — a System-1 typed-decision server on the
      # 3090, sibling to services.laya-serve / services.intern-decision in
      # missionctrl-infra. Checkpoint downloads from HF into the state dir on
      # first request.
      nixosModules.default = { config, lib, pkgs, ... }:
        let cfg = config.services.vegaml;
        in {
          options.services.vegaml = {
            enable = lib.mkEnableOption "vegaml typed-decision server";
            package = lib.mkOption {
              type = lib.types.package;
              default = self.packages.${pkgs.stdenv.hostPlatform.system}.vegaml-serve;
              description = "The vegaml-serve wrapper (CUDA cu128 venv).";
            };
            host = lib.mkOption { type = lib.types.str; default = "0.0.0.0"; };
            port = lib.mkOption { type = lib.types.port; default = 8200; };
            model = lib.mkOption {
              type = lib.types.str;
              default = "4b";
              description = ''VEGAML_MODEL: "" = 0.8B, "4b" = 4B.'';
            };
            openFirewall = lib.mkOption { type = lib.types.bool; default = false; };
          };
          config = lib.mkIf cfg.enable {
            networking.firewall.allowedTCPPorts = lib.mkIf cfg.openFirewall [ cfg.port ];
            systemd.services.vegaml = {
              description = "vegaml typed-decision server (CUDA)";
              wantedBy = [ "multi-user.target" ];
              after = [ "network-online.target" ];
              wants = [ "network-online.target" ];
              environment = {
                VEGAML_MODEL = cfg.model;
                HF_HOME = "/var/lib/vegaml/hf";
              };
              serviceConfig = {
                ExecStart = "${cfg.package}/bin/vegaml-serve --host ${cfg.host} --port ${toString cfg.port}"
                  + lib.optionalString (cfg.model != "") " --model ${cfg.model}";
                DynamicUser = true;
                StateDirectory = "vegaml";
                Restart = "on-failure";
                RestartSec = 5;
              };
            };
          };
        };
    };
}
