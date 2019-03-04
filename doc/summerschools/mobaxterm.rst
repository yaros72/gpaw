.. _mobaxterm:

===========================
MobaXTerm for Windows users
===========================

**These instructions are for Windows users only.**


Install MobaXterm
=================

Go to the `MobaXterm Homepage <http://mobaxterm.mobatek.net/>`_ and click "Get MobaXTerm Now".  Choose the free Home edition.  You can choose between a portable version and an installable version, we have only tested the portable but expect that both work.  Download it.  Unpack the ZIP file.  You can now move the executable to whereever you want, e.g. your desktop or onto a USB stick so you can use MobaXterm from other Windows machines.


Logging in the first time
=========================

When you run the program the first time you just need to write the name of server in the ``Quick Connect`` field in the upper right corner, then press enter.  The server name is ``login.gbar.dtu.dk``.  

A new window opens.  Choose the connection type SSH on the top bar.  Specify your DTU user name in the corresponding field.  A terminal tab now opens in the original MobaXTerm window, and you are asked to type the password.

Logging in next time
====================

MobeXterm remembers your settings.  Just click on ``login.gbar.dtu.dk`` under Saved Sessions.

When you are logged in
======================

You now have a window on the GBAR login-computer.  To proceed to an interactive compute node, use the command::

  linuxsh -X

Please note that MobaXTerm also gives access to the files on the server, so you can transfer files to your own computer if you so desire.


