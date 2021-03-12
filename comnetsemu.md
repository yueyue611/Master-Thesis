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
  
    - **ERROR 1:** *configure: error: You must install the glib-2.0 gobject-2.0 >= 2.48.0 pkg-config module to compile libvirt*
  
    ![libvirt3](https://user-images.githubusercontent.com/39553089/110870661-32602380-82cd-11eb-96b1-6324f198a61f.png)
    ![libvirt4](https://user-images.githubusercontent.com/39553089/110871055-f2e60700-82cd-11eb-99c3-b59c52e72044.png)
  
    - **ERROR 2:** *configure: error: You must install the gnutls >= 3.2.0 pkg-config module to compile libvirt*
  
    ![libvirt5](https://user-images.githubusercontent.com/39553089/110871065-f8435180-82cd-11eb-9739-78423b17ad41.png)
    ![libvirt6](https://user-images.githubusercontent.com/39553089/110873740-b5d04380-82d2-11eb-9c3f-3c076222b173.png)
  
    - **ERROR 3:** *configure: error: libnl3-devel is required for macvtap support*
  
    ![libvirt7](https://user-images.githubusercontent.com/39553089/110873886-f5972b00-82d2-11eb-8007-65a85e3a6505.png)
    ![libvirt8](https://user-images.githubusercontent.com/39553089/110874487-18760f00-82d4-11eb-80f8-36750e56936b.png)
  
    - **ERROR 4:** *configure: error: Package requirements (libnl-route-3.0) were not met*
  
    ![libvirt9](https://user-images.githubusercontent.com/39553089/110874501-1e6bf000-82d4-11eb-861d-5b238af07115.png)
    ![libvirt10](https://user-images.githubusercontent.com/39553089/110874663-74d92e80-82d4-11eb-83cd-381ae4466fad.png)
  
    - **ERROR 5:** *configure: error: libxml2 >= 2.9.1 is required for libvirt*
  
    ![libvirt11](https://user-images.githubusercontent.com/39553089/110874871-d5686b80-82d4-11eb-9513-d012e3e3fcbd.png)
    ![libvirt12](https://user-images.githubusercontent.com/39553089/110874881-da2d1f80-82d4-11eb-8e6f-477143d19411.png)
  
    - **ERROR 6:** *configure: error: "xsltproc is required to build libvirt"*
  
    ![libvirt13](https://user-images.githubusercontent.com/39553089/110875462-f3829b80-82d5-11eb-8089-117ab95de8d1.png)
    ![libvirt14](https://user-images.githubusercontent.com/39553089/110875469-f7162280-82d5-11eb-91eb-3f37a83540a1.png)
  
    - **ERROR 7:** *configure: error: You must install device-mapper-devel/libdevmapper to compile libvirt with mpath storage driver*
  
    ![libvirt15](https://user-images.githubusercontent.com/39553089/110875807-9a673780-82d6-11eb-847b-e89cc09bb52f.png)
    ![libvirt16](https://user-images.githubusercontent.com/39553089/110875815-9e935500-82d6-11eb-98cf-5034075a2e45.png)
  
    - **ERROR 8:** *configure: error: You must install the pciaccess module to build with udev*
  
    ![libvirt17](https://user-images.githubusercontent.com/39553089/110876342-91c33100-82d7-11eb-907d-1205be689080.png)
    ![libvirt18](https://user-images.githubusercontent.com/39553089/110876353-95ef4e80-82d7-11eb-95e5-48c5e3668c17.png)
  
    - **Warning:**
    ![libvirt19](https://user-images.githubusercontent.com/39553089/110876361-9ee02000-82d7-11eb-9bc4-9d5fc5a68a9a.png)
  
  Afterwards:
  
  ```
  ~/build$ make
  ~/build$ sudo make install
  ```
  
  ![libvirt20](https://user-images.githubusercontent.com/39553089/110877606-ec5d8c80-82d9-11eb-80d2-21d0b3b36cd4.png)
  ![libvirt21](https://user-images.githubusercontent.com/39553089/110877614-f1bad700-82d9-11eb-8143-d9a4c8e0e808.png)
  
  
## 3. Install the plugin

- Check the [guide](https://github.com/vagrant-libvirt/vagrant-libvirt#installation)

  ```
  sudo apt-get build-dep vagrant ruby-libvirt
  sudo apt-get install qemu libvirt-bin ebtables dnsmasq-base
  sudo apt-get install libxslt-dev libxml2-dev libvirt-dev zlib1g-dev ruby-dev
  ```

  **ERROR:** *E: You must put some 'source' URIs in your sources.list*

  ![plugin1](https://user-images.githubusercontent.com/39553089/110879153-c1c10300-82dc-11eb-9600-0efe673f33bf.png)

  Click *Source Code* and reload it.
  
  ![plugin2](https://user-images.githubusercontent.com/39553089/110879283-ff259080-82dc-11eb-8a36-912af68f6bcf.png)
  ![plugin3](https://user-images.githubusercontent.com/39553089/110879352-211f1300-82dd-11eb-9d78-4b663b8b4611.png)
  


  



  
  
  
  
  
  
  
  
  





  
  


  





  
  



  




  
  

- 




		
