.. _setupwin:

===================================
Setting up the first time (Windows)
===================================

You will need the program MobaXterm which will help you with the
following tasks

  * Logging in to the databar.

  * Displaying the ASE graphical user interface on your laptop.

  * Forward a connection to the Jupyter server (SSH tunnel).

  * Accessing files on the remote server from your laptop.


Installing MobaXterm
====================

You download the program from the website
https://mobaxterm.mobatek.net.  Choose Download, select then "Free
Home Edition".  You are now given a choice between a "Portable
Edition" and an "Installer Edition".  The Installer Edition is
installed like any other Windows program; the Portable Edition comes
as a ZIP file that you need to unpack.  The program is then in a
folder together with a data file, and you can run it from this
folder.  We have no reason to recommend one version over the other, it
is a matter of personal taste.


Connecting the first time
=========================

Start MobaXterm.  You will see a window with a row of buttons at the
top.  Click on the *Session* button, you will now see a window as
shown below.

.. image:: Moba_ssh.png
   :width: 66%

In the tab *Basic SSH settings* you should choose *Remote host* to be
``login.gbar.dtu.dk``.  The user name is your DTU user name (external
participants got it in the registration material).  The port number
must remain 22.  Click *OK*  and give your DTU password in the text
window when prompted.  **NOTE** Nothing is written when you type the
password, not even stars.

**We do not recommend allowing MobaXterm to remember your password!**

You now have a command-line window on the DTU login-computer, as shown
below.

.. image:: Logged_in_Win.png
   :width: 66%

The two last lines are the command prompt.  The first line indicates
your current working directory, here your home folder symbolized by
the ~ (tilde) character.  The lower line gives the name of the
computer (``gbarlogin``) and the user name (``jasc`` in the figure)
followed by a dollar sign.

This computer (``gbarlogin``) may not be used to calculations, as it
would be overloaded.  You therefore need to log in to the least loaded
interactive computer by writing the command::

  linuxsh -X

(the last X is a capital X, you get no error message if you type it
wrong, but the ASE graphical user interface will not work).


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
<accesswin>`

