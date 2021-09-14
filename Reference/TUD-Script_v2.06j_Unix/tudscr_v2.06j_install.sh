#!/bin/bash
#
# Autor: Falk Hanisch, Jons-Tobias Wamhoff
#
# getestet auf:
# Ubuntu 14.04
# Ubuntu 15.04
#
# in Kombination mit:
# TeX Live 2016
#
# Notwendige Tools:
# unzip        (Ubuntu package unzip)
#
# Vorausgesetzte LaTeX Pakete:
# fontinst (Ubuntu package texlive-font-utils)
# lmodern  (Ubuntu package lmodern)
# cm-super (Ubuntu package cm-super)
# cmbright (Ubuntu package texlive-fonts-extra)
# hfbright (Ubuntu package texlive-fonts-extra)
# iwona    (Ubuntu package texlive-fonts-extra)
#
# Benoetigte Archive (im Verzeichnis des Installationsskriptes):
# DIN_Bd_PS.zip
# Univers_PS.zip
#
# Die Installation erfolgt in Normalfall in das lokale Benutzerverzeichnis
# $TEXMFHOME. Dieses entspricht unter Linux in '~/texmf' und unter Mac OS in
# '~/Library/texmf'. Wird das Skript mit 'sudo' ausgefuehrt, erfolgt die
# systemweite Installation fuer alle Nutzer in $TEXMFLOCAL.
#
SOURCE="${BASH_SOURCE[0]}"
# if $SOURCE was a relative symlink, we need to resolve it relative to the path
# where the symlink file was located
while [ -h "$SOURCE" ]; do
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
echo
echo  =====================================================================
echo
echo   $DIR
echo
echo  =====================================================================
echo
echo   Installation TUD-Script unter Unix
echo     2020/09/29 v2.06j TUD-Script
echo
checkfile()
{
  if [ ! -f "$1" ] ; then
    missing=true
    missingfile "$1"
  else
    echo   Datei $1 gefunden.
  fi
}
missingfile()
{
  echo  =====================================================================
  echo
  echo   Die Datei $1 wurde nicht gefunden. Diese wird fuer die
  echo   Installation zwingend benoetigt. Bitte kopieren Sie $1
  echo   in das Verzeichnis des Skriptes und fuehren dieses abermals aus.
  echo   Beachten Sie, dass die Schriftarchive speziell fuer die Verwendung
  echo   von LaTeX bestellt werden muessen, ein alleiniges Umbenennen
  echo   falscher Archive wird zu Fehlern bei der Installation fuehren.
  echo
  echo  =====================================================================
}
checkpackage()
{
  package=$(kpsewhich $1)
  if [ -z "$package" ] ; then
    missing=true
    missingpackage "$1" "$2"
  else
    echo   Paket $2 \($1\) gefunden.
  fi
}
missingpackage()
{
  echo  =====================================================================
  echo
  echo   Das LaTeX-Paket $2 \($1\) wurde nicht gefunden.
  echo   Dieses wird fuer die Schriftinstallation zwingend benoetigt.
  echo   Bitte das Paket \'$2\' ueber die Distribution installieren und
  echo   danach dieses Skript abermals ausfuehren.
  echo
  echo  =====================================================================
}
checkscript()
{
  script=$(find $texpath -name "$1")
  if [ -z "$script" ] ; then
    missing=true
    missingscript "$1" "$2"
  else
    echo   Skript $1 aus Paket $2 gefunden.
  fi
}
missingscript()
{
  echo  =====================================================================
  echo
  echo   Das ausfuehrbare Skript $1 aus dem Paket $2 wurde nicht
  echo   gefunden. Dieses wird im Normalfall von der LaTeX-Distribution
  echo   bereitgestellt und zur Schriftinstallation zwingend benoetigt.
  echo   Bitte das Paket \'$2\' ueber die Distribution installieren und
  echo   danach dieses Skript abermals ausfuehren.
  echo
  echo  =====================================================================
}
proof_userinput()
{
  echo
  echo  =====================================================================
  echo
  echo   $texmfpath
  echo
  echo   Soll dieser Pfad genutzt werden?
  if [ ! -d $texmfpath ] ; then
    echo   Der angegebene Ordner existiert nicht, wird jedoch erstellt.
  fi
  select yn in "Ja (empfohlen)" "Nein"; do
    case $yn in
      "Ja (empfohlen)") break;;
      "Nein")
        set_texmfpath
        break;;
    esac
  done
}
set_texmfpath()
{
  echo
  echo   Geben Sie das Installationsverzeichnis an:
  read texmfpath
  proof_userinput
}
mkvaldir()
{
  mkdir -p $1
  if [ $? -ne 0 ] ; then
    echo Keine Schreibberechtigung fuer folgenden Pfad:
    echo $1
    echo Versuchen Sie das Ausfuehren mit \'sudo -k bash <Skriptname>\'
    abort
  fi
}
abort()
{
  echo
  echo  =====================================================================
  echo   Abbruch der Installation, temporaere Dateien werden geloescht.
  echo  =====================================================================
  echo
  read -n1 -r -p "Druecken Sie eine beliebige Taste . . . "
  echo
  rm -rf tudscrtemp
  exit 0
}
symlinkpath=$(which tex)
while [ -h "$symlinkpath" ]; do
  DIR="$( cd -P "$( dirname "$symlinkpath" )" && pwd )"
  symlinkpath="$(readlink "$symlinkpath")"
  [[ $symlinkpath != /* ]] && SOURCE="$texpath/$symlinkpath"
done
texpath="$( cd -P "$( dirname "$symlinkpath" )" && pwd )"
if [ -z "$texpath" ] ; then
  echo Es wurde keine LaTeX-Distribution gefunden.
  echo Moeglicherweise hilft der Aufruf des Skriptes mit:
  echo "'sudo -k env \"PATH=\$PATH\" bash $0'"
  abort
else
  PATH=$texpath:$PATH
fi
echo
echo   LaTeX-Distribution gefunden in:
echo   \'$texpath\'
echo
echo  =====================================================================
echo
rm -rf tudscrtemp
mkvaldir tudscrtemp/converted
echo  =====================================================================
echo
echo   Notwendige Dateien und Pakete werden gesucht.
echo   Dies kann einen Moment dauern.
echo
missing=false
version="$(basename $0)"
version=$(echo $version|cut -c8-)
version=$(echo $version|rev|cut -c12-|rev)
checkfile "tudscr_$version.zip"
if $missing ; then
  abort
fi
echo
echo   Es wurden alle notwendigen Dateien und Pakete gefunden.
echo
echo  =====================================================================
echo  =====================================================================
echo
if [ "$EUID" -eq 0 ] ; then
  texmfpath=$(kpsewhich --var-value=TEXMFLOCAL)
  echo   Mehrbenutzerinstallation \(Administrator\).
else
  texmfpath=$(kpsewhich --var-value=TEXMFHOME)
  echo   Einzelbenutzerinstallation.
fi
echo
echo  =====================================================================
echo  =====================================================================
echo
echo   Bitte geben Sie das gewuenschte Installationsverzeichnis an.
echo   Dieses sollte sich jenseits der Distributionsordnerstruktur
if [ "$EUID" -eq 0 ] ; then
  echo   in einem Pfad mit Lese-Zugriff fuer alle Benutzer befinden.
else
  echo   in einem lokalen Benutzerpfad befinden.
fi
echo
echo  =====================================================================
echo   Sie sollten nachfolgend den eingestellten Standardpfad verwenden.
echo   Aendern Sie diesen nur, wenn Sie genau wissen, was Sie tun.
proof_userinput
echo
echo   Installation in folgenden Pfad:
echo   $texmfpath
echo  =====================================================================
echo
unzip -o tudscr_$version.zip -d $texmfpath
texhash
echo
echo  =====================================================================
echo   Die Installation wird beendet.
echo   Der Ordner mitsamt aller temporaeren Dateien wird geloescht.
echo  =====================================================================
echo   Dokumentation und Beispiele fuer das TUD-Script-Bundle sind
echo   unter $texmfpath/doc/latex/tudscr oder
echo   ueber den Terminalaufruf \'texdoc tudscr\' zu finden.
echo  =====================================================================
cd ../..
read -n1 -r -p "Druecken Sie eine beliebige Taste . . . "
echo
rm -rf tudscrtemp
exit 0
