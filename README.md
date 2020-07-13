# Rendy Sphere Visualizer

Rendy Sphere Visualizer is a render optimized for rendering spheres it uses different technics like distance field ambient occlusion and raytraced shadows to archive high fidelity graphics in real-time. I mainly use it as a visualizer for music. As the name implies I have used the Rendy rendering framework. Rendy uses gfx-hal which is a hardware abstraction layer that abstracts many graphics APIs like Vulkan, D3D, Metal, or OpenGL. Therefore, this project can run on a variety of platforms.

[![video](http://img.youtube.com/vi/Hfbo6E0vXDM/0.jpg)](http://www.youtube.com/watch?v=Hfbo6E0vXDM "video")

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Please go to [rustup.rs](https://rustup.rs/) and follow the 
instructions. It will be quick and painless (hopefully).

This repository uses [git-lfs](https://git-lfs.github.com/). 
Please go ahead and download and install git-lfs. Please
make sure after you have cloned the repository, that you
pulled the lfs-file with the following command.

```
git lfs pull
```

### Installing

If you have installed rustup and successfully cloned the 
repository, you can go ahead open a terminal in the project
folder and run: 

```
// Linux
cargo run --features vulkan --release -- AUDIO_FILE[.mp3|.wav|.ogg]

// Windows
cargo run --features dx12 --release -- AUDIO_FILE[.mp3|.wav|.ogg]

// Mac Os
cargo run --features metal --release -- AUDIO_FILE[.mp3|.wav|.ogg]
```

If you have a problem running the appropriate Command on the
the operating system of your choice fears not opening an issue. 
I do not have all the operating systems at my disposal to test
all the backends.

## Built With

* [Rust](https://www.rust-lang.org/) - The programming language
* [Cargo](https://doc.rust-lang.org/cargo/) - Dependency Management
* [Rendy](https://github.com/amethyst/rendy) - The rendering framework
* [Legion](https://github.com/TomGillen/legion) - entity component system
* [NPhysics](https://nphysics.org/) - physics engine
* [Rodio](https://github.com/RustAudio/rodio) - audio playback library
* [Nalgebra](https://nalgebra.org/) - The linalg library
* [Serde](https://serde.rs/) - The serialization library

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


