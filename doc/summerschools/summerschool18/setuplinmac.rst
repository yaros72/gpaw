.. _setuplinmac:

=========================================
Setting up the first time (Linux / macOS)
=========================================

You will be using the command line in a Terminal to access the DTU
cental computers, and to set up the connections you need to display
the output on your laptop.


Mac users: Install XQuartz
==========================

Linux users should skip this step as an X11 server is part of all
nornal Linux distributions.

As a Mac user, you should install an X11 server, it is needed to
display the ASE Graphical User Interface on your laptop.  If you do
not install it, you can still run Jupyter Notebooks, but the command
to view structures with the GUI will not work.

Go to https://www.xquartz.org/ and download the newest version of
XQuartz.  It is an open-source port of the X11 windows system for
macOS, it used to be part of macOS until version 10.7, but was then
removed.  Although it is no longer an official part of macOS, it is
still Apple developpers that maintain the project.

After installing it, you will have to log out of your Mac and log in
again.



Open two Terminal windows
=========================

You will need two open terminal windows, where you can write Unix
commands directly to the operating system.

macOS
   The program is called ``Terminal`` and is in the folder ``Other``
   in Launchpad.  You can also find it in Spotlight by searching for
   Terminal.

Linux
   The name and placement of the program depends on the distribution.
   It is typically called ``Terminal``,  ``LXterminal``, ``xterm`` or
   something similar.  Search for Terminal in spotlight, or look in
   menus named "System tools" or similar.

Once you have opened a window you can typically get a new one by
pressing Cmd-N (Mac) or Crtl-Shift-N (Linux).


Log into the databar
====================

You use the ``ssh`` (Secure SHell) command to create a secure
(encrypted) connection to the databar computers.  In the terminal,
write::

  ssh -XY USERNAME@login.gbar.dtu.dk

where ``USERNAME`` is your DTU user name (external participants got it
in their registration material).  Note the ``-XY`` option, it is a minus
followed by a capital X and a capital Y, it tells ssh to let the
remote computer open windows on your screen.

Note that when you write your DTU password, you cannot see what you
type (not even stars or similar!).


You now have a command-line window on the DTU login-computer.  This
computer (``gbarlogin`` a.k.a. ``login.gbar.dtu.dk``) may not be used
to calculations, as it would be overloaded.  You therefore need to log
in to the least loaded interactive computer by writing the command::

  linuxsh -X

You now have a command-line window on an interactive compute node, as shown
below.

.. image:: Logged_in_Mac.png
   :width: 66%

The two last lines are the command prompt.  The first line indicates
your current working directory, here your home folder symbolized by
the ~ (tilde) character.  The lower line gives the name of the
computer (here ``n-62-27-23``) and the user name (``jasc`` in the figure)
followed by a dollar sign.



Get access to the software
==========================

To give access to the software you need for this course, please run
the command::

  source ~jasc/setup2018

Note the tilde in the beginning of the second word.

The script give you access to ASE, GPAW and related software.  It will
install Jupyter Notebook in your own account (necessary as the
visualization will otherwise not work).

The script will ask you to **set a Jupyter Notebook password.** This
will be used to access the notebooks from the browser on your laptop.
It is a bad idea to type your DTU password into untrusted programs, so
you should probably choose a different password - *this is
particularly important if you are a DTU student/employee, the security
of your DTU password is critical!*

The script will also copy a selection of draft notebooks to a folder
called CAMD2018 in your DTU databar account.

	   
Carrying on
===========

Now read the guide for :ref:`Starting and accessing a Jupyter Notebook
<accesslinmac>`

