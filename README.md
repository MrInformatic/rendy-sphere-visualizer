# Rendy Sphere Visualizer

Rendy Sphere Visualizer is a render optimized for rendering 
spheres it uses different like distance field ambient occlusion
and raytraced shadows to archive high fidelity graphics in
real time. I manly use it as an visualizer for music. As the 
name implies I have used the Rendy rendering
framework. Rendy uses gfx-hal which is an hardware abstraction
layer which abstract many graphics apis like Vulkan, D3D, Metal
or OpenGL. Therefore, this project can run on a variety of 
platforms.

[![video](http://img.youtube.com/vi/Hfbo6E0vXDM/0.jpg)](http://www.youtube.com/watch?v=Hfbo6E0vXDM "video")

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Pleas go to [rustup.rs](https://rustup.rs/) and follow the 
instructions. It will be quick and painless (hopefully).

### Installing

If you have installed rustup and successfully cloned the 
repository, you can go ahead open a terminal in the project
folder and run: 

```
// Linux
cargo run --features vulkan --release

// Windows
cargo run --features dx12 --release

// Mac Os
cargo run --features metal --release
```

If you have a problem running the appropriate Command on the
operating system of your choice fear not opening an issue. 
I do not have all the operating systems at my disposal to test
all the backends.

## Built With

* [Rust](https://www.rust-lang.org/) - The programming language
* [Cargo](https://doc.rust-lang.org/cargo/) - Dependency Management
* [Rendy](https://github.com/amethyst/rendy) - The rendering framework
* [Nalgebra](https://nalgebra.org/) - The linalg libary
* [Serde](https://serde.rs/) - The serialization libary

For a complete list see the Cargo.toml file under dependencies.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/MrInformatic/rendy-sphere-visualizer/tags). 

## Author

* **Philipp Haustein** - [MrInformatic](https://github.com/MrInformatic)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* [rendy-pbr](https://github.com/termhn/rendy-pbr) for providing
examples on how to use Rendy


