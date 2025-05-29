{
  description = "Train neural networks that distill into logic circuits, using JAX";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, utils }:
    utils.lib.eachDefaultSystem (system:
      with import nixpkgs { inherit system; }; {
        defaultPackage = stdenv.mkDerivation {
          name = "difflogic";
          src = ./.;
          nativeBuildInputs = [ (python3.withPackages (p: with p; [ p.jax p.einops p.optax ])) ];
          buildPhase = ''
            python main.py
            ${stdenv.cc.targetPrefix}cc $NIX_CFLAGS_COMPILE gate.c -o gate
          '';
          installPhase = ''
            install -Dm755 -t $out/bin gate
          '';
          meta.mainProgram = "gate";
        };
      }
    );
}
