.. _summerschool16:

=======================
CAMd Summer School 2016
=======================

Announcement:
http://www.fysik.dtu.dk/english/Research/CAMD/Events/Summer-school-2016

.. highlight:: bash


Logging in to the databar
=========================

The software (mainly Python, ASE and GPAW) that you need for the
computer exercises are available in the databar.  You are expected to
access the databar from your own laptops.  There are essentially two
ways for doing that: ThinLinc and Secure Shell.  ThinLinc will give
you a Linux desktop, Secure Shell will open windows on your normal
desktop.

:Linux and Mac users:
  We recommend using Secure Shell.

:Windows users:
  ThinLinc is easier to install, many find that Secure Shell is nicer
  to work with.


Using Secure Shell on Linux and Mac computers
---------------------------------------------

..

  **Mac users**: You need to install the semi-official X-server for
  MacOS: http://xquartz.macosforge.org/

To log in, open a Terminal window.  On Ubuntu Linux click on the Dash Home
and search for Terminal - we recommend dragging it to the dock.  In Mac OS X,
find it in Spotlight, again we suggest dragging it to the dock.

In the terminal window, type::

    $ ssh -X login.gbar.dtu.dk

and once you are logged in, proceed to log in to one of the Linux machines
with::

    $ linuxsh -X

Note that first you log in to a login server that cannot run our
software, the second command then logs you on to one of the least
loaded Linux servers where the software is expected to
work. **Forgetting to run the ``linuxsh -X`` command every time you
login is the most common source of errors!**

You now need to read `Setting up your UNIX environment`_.


Installing and using Secure Shell on Windows computers
------------------------------------------------------

To log in to the databar and display the applications on your Windows
desktop, you need to install an X11 server on your Windows machine.
We recommend installing `MobaXterm <http://mobaxterm.mobatek.net/>`_.
See also :ref:`mobaxterm`.

The server name is login.gbar.dtu.dk

Once you have an xterm terminal window open on the gbar login server, type::

    $ linuxsh -X

to proceed to one of the Linux servers, where the course software is installed.

You now need to read `Setting up your UNIX environment`_.


Installing and using ThinLinc
-----------------------------

**This is an alternative way to access the computers from Macs and
Windows machines, if for some reasons you do not wish to use SSH.**

Information on how to install and use ThinLinc is available here:
http://gbar.dtu.dk/index.php/faq/43-thinlinc

Set the server name to thinlinc.gbar.dtu.dk. User name and password is
your DTU login.  When loggin in, you are asked to choose between
different desktops, we recommend choosing Xfce (some of the other
choices will cause trouble).

You need to open a terminal window to use gpaw.  Click on the
Applications Menu, then on Terminal Emulator.

*Hint*: Before you log in the first time, click on Options, choose the
Screen tab, and select "Work area (maximized)".  Then thinlinc will
open a window filling the whole screen, but will not go into
full-screen mode which many people find annoying to get out of again.


Setting up your UNIX environment
================================

The first time you use the databar computers, you must configure your
environment.  Add the following line to the very end of your
``.bashrc`` file::

    source ~mikst/camd2016.bashrc


This will set up the environment for you so that you can use ASE, GPAW and
matplotlib.

Note that the file ``.bashrc`` starts with a period, making it a hidden file in Unix.

If you would like to setup the text editors ``nedit`` and ``vim`` in a python
friendly way you can execute the command::

    $ sh ~mikst/python_friendly.sh

This may take about a minute or so.


Running GPAW calculations
=========================

**Warning** do not use spaces in the directory/file names!

GPAW calculations are written as Python scripts, which can be run with
the command::

    $ python filename.py

If the calculation lasts more than a few seconds, submit it to the
queue instead of running it directly::

    $ gpaw-qsub filename.py

This will allow the script to be executed on a different host, so the
jobs will be distributed efficiently even if many users logged on to
the same computer.  The output from your script will be written to the
files :samp:`filename.py.oNNNNNN` and :samp:`filename.py.eNNNNNN`
where :samp:`NNNNNN` is a job number.  Normal output (stdout) goes to
the :samp:`.oNNNNNN` file, whereas error messages (stderr) goes to
:samp:`.eNNNNNN`.  Unlike some queue systems these files appear when
the job starts, so you can follow job progress by looking at them.

You can run jobs in parallel, using more CPUs for
increased speed, by specifying e.g. 4 CPUs like this::

    $ gpaw-qsub -p 4 filename.py

The ``qstat`` or :samp:`qstat -u {USERNAME}` commands can be used to
monitor running jobs, and :samp:`qdel {JOB_ID}` to delete jobs if
necessary.  On the joblist from ``qstat``, you can find the JOB_ID.
You can also see the status of the jobs, Q means queued, R means
running, C means completed (jobs remain on the list for a while after
completing).


Exercises and Tutorials
=======================

You are now ready to embark on the :ref:`exercises` and :ref:`tutorials`.
Have fun.


Accessing databar files on your laptop
======================================

The databar has its own file system, separate from your laptop
(obviously).  If you want to keep your scripts and/or results, the
files should be transferred to your own computer.

* If you use MobaXTerm on Windows, you can access your databar files
  by dragging and dropping in the file window to the left.

* If you use ssh to log in, and are familiar with the scp command, you
  can use it to copy files back and forth::

    scp camd16XXX@login.gbar.dtu.dk:myfolder/myfile .
    scp localfile camd16XXX@login.gbar.dtu.dk:myfolder

* You can map your databar home folder as a network drive:

  - **On a MAC**: From Finder select Go / Connect to Server (⌘K) and
    enter the following as the Server Address::

      smb://home.cc.dtu.dk/camd16XXX

  - **On Windows**: Tools | Map Network drive

    And select any unused drive letter and enter the following as path::

      \\home.cc.dtu.dk\camd16XXX

  - **On Linux**: It depends on the Linux distribution and on what
    software is installed.  If you have samba installed, you should be
    able to mount the drive using the smb protocol, use the same
    address as on a Mac (see above).

    Another option that may work is the sshfs command. Create an empty
    directory, and mount the remove file system there::

      mkdir databar
      sshfs camd16XXX@transfer.gbar.dtu.dk: databar

    If you fumble, you may leave the folder in a half-mounted state,
    and get a "Device or resource not ready" or "Transport endpoint is
    not connected" error when you try again.  In those case, either create a new folder
    with a new name, or reboot your linux machine.

Cannot open new windows after 20 minutes
========================================

If you use SSH to log in from a Mac (and from *some* Linux
distributions), an over-zealous security measure prevents the databar
computer from opening new windows if it is more than 20 minutes since
you logged in.

To fix it, edit the file /etc/ssh/ssh_config as the super-user, and
add a magic line to it::

  cd /etc/ssh
  sudo nano ssh_config

This opens a primitive editor.  Use the arrow keys to go to the end of
the file, and add this line::

  ForwardX11Timeout 36000

(it is case sensitive, so use copy-paste).

Save the file and exit (Control-X).  This increases the time-out to 36000 seconds, i.e. ten hours.


Notes and hints
===============

* Editor: Several editors are available including emacs, vim and gedit.

* Printer: There is a printer in each databar, the name is written on
  the printer. To use it from a terminal: :samp:`lp -d {printename}
  {filename}`.  Read more about printing `here
  <http://www.gbar.dtu.dk/wiki/Printing>`_.

* To open a pdf-file: :samp:`evince {filename.pdf}`

* The normal tilde (~) key combination is not functional on the
  databar computers.  Use :kbd:`Alt Graph + 5` to type a tilde.

.. * How to `use USB sticks <http://www.gbar.dtu.dk/wiki/USB_Access>`_.
