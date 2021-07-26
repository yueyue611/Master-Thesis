# Using Git
see common Git commands in [Using Git](https://docs.github.com/en/github/using-git)

# Basic writing and formatting syntax
see basic uses in [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/basic-writing-and-formatting-syntax)

# Add SSH key
## 1. Generating a new SSH key and adding it to the ssh-agent

![Add SSH key 1](https://user-images.githubusercontent.com/39553089/110380951-20764a80-8059-11eb-8730-ef386a3f215c.png)

## 2. Adding a new SSH key to your GitHub account

![Add SSH key 2](https://user-images.githubusercontent.com/39553089/110381015-371ca180-8059-11eb-9fd9-0081cd12fae8.png)

## 3. Switching remote URLs from HTTPS to SSH

![Add SSH key 3](https://user-images.githubusercontent.com/39553089/110381036-3e43af80-8059-11eb-994c-3a6c0c1219ed.png)

## 4. Creating branches and a pull request, updating a feature branch

[Creating a branch](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-and-deleting-branches-within-your-repository)

```
$ git branch try-branch
```

[Creating a pull request](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)

```
$ git checkout try-branch
$ $ git add .
$ git status
$ git commit -m "comment"
$ git push origin try-branch
```

[Updating a feature branch](https://gist.github.com/whoisryosuke/36b3b41e738394170b9a7c230665e6b9)

```
$ git checkout main
```

```
$ git fetch origin
$ git merge origin/main

or just

$ git pull origin main
```

```
$ git checkout try-branch
$ git merge main
$ git push origin try-branch
```

