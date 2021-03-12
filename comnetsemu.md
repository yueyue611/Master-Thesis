# [Option 1: Install in a Vagrant managed VM (Highly Recommended)](https://git.comnets.net/public-repo/comnetsemu#option-1-install-in-a-vagrant-managed-vm-highly-recommended)

## 1. Install Vagrant: v2.2.5
- [Download address](https://releases.hashicorp.com/vagrant/2.2.5/), [Quick Start](https://learn.hashicorp.com/tutorials/vagrant/getting-started-index?in=vagrant/getting-started)

- Download the Vagrant package, and then install it after downloading the .deb file

![vagrant1](https://user-images.githubusercontent.com/39553089/110852563-d12b5680-82b2-11eb-94c7-2849afe99ddd.png)

- To verify the installation, check the version

![vagrant2](https://user-images.githubusercontent.com/39553089/110852667-f0c27f00-82b2-11eb-930b-08f16fb1d482.png)

- Create a directory

`~$ mkdir vagrant_getting_started`

`~$ cd vagrant_getting_started`

- Initialize the directory and specify the *hashicorp/bionic64* box

`vagrant init hashicorp/bionic64`

  You have now initialized your first project directory and have a *Vagrantfile* in your current directory. To set up Vagrant for an existing project you would run *vagrant init* in a pre-existing directory.
 
 - Find More Boxes
 
   The best place to find more boxes is [HashiCorp's Vagrant Cloud box catalog](https://app.vagrantup.com/boxes/search). 
   
 - 


## 2. Install  Libvirt: v5.10.0
- [Download address](https://libvirt.org/sources/)

  Choose *libvirt-5.10.0.tar.xz*, download it and move it to the directory *~/home/gaoyueyue*(~$). Instructions on building and installing libvirt can be found on the [website](https://libvirt.org/compiling.html).
  
  ```
  ~$ xz -dc libvirt-5.10.0.tar.xz | tar xvf -
  ~$ cd libvirt-5.10.0
  ~/libvirt-5.10.0$ meson build
  ````
  
  ![libvirt1](https://user-images.githubusercontent.com/39553089/110870613-18bedc00-82cd-11eb-9f0c-29a6d6a151f0.png)

  
  **ERROR!** Then read *README* and follow the instructions.
  
  ```
  ~$ mkdir build && cd build
  ~/build$ ../configure --prefix=/usr --sysconfdir=/etc --localstatedir=/var
  ```
  
  ![libvirt2](https://user-images.githubusercontent.com/39553089/110870650-2d02d900-82cd-11eb-8cb4-5f6ade128658.png)
  
  **ERROR!** Then install the dependencies.
  
  *configure: error: You must install the glib-2.0 gobject-2.0 >= 2.48.0 pkg-config module to compile libvirt*
  
  ![libvirt3](https://user-images.githubusercontent.com/39553089/110870661-32602380-82cd-11eb-96b1-6324f198a61f.png)
  ![libvirt4](https://user-images.githubusercontent.com/39553089/110871055-f2e60700-82cd-11eb-99c3-b59c52e72044.png)
  
  *configure: error: You must install the gnutls >= 3.2.0 pkg-config module to compile libvirt*
  
  ![libvirt5](https://user-images.githubusercontent.com/39553089/110871065-f8435180-82cd-11eb-9739-78423b17ad41.png)
  ![libvirt6](https://user-images.githubusercontent.com/39553089/110873740-b5d04380-82d2-11eb-9c3f-3c076222b173.png)
  
  *configure: error: libnl3-devel is required for macvtap support*
  
  ![libvirt7](https://user-images.githubusercontent.com/39553089/110873886-f5972b00-82d2-11eb-8007-65a85e3a6505.png)
  



  




  
  

- 




		
