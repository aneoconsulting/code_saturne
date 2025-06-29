## Process this file with automake to produce Makefile.in

#-------------------------------------------------------------------------------

# This file is part of code_saturne, a general-purpose CFD tool.
#
# Copyright (C) 1998-2025 EDF S.A.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51 Franklin
# Street, Fifth Floor, Boston, MA 02110-1301, USA.

#-------------------------------------------------------------------------------

# Main LaTeX document

TEXMAIN = theory.tex

# Other LaTex documents

TEXOTHERS = \
algebr.tex \
apintr.tex \
atmo.tex \
cs_balance.tex \
bndcnd.tex \
cavitation.tex \
cs_turbulence_ke.tex \
cfbase.tex \
cfener.tex \
cfmsvl.tex \
cfqdmv.tex \
cfxtcl.tex \
clptrg.tex \
cs_boundary_conditions_set_coeffs_turb.tex \
clsyvt.tex \
cs_equation_iterative_solve.tex \
combustion_coal.tex \
combustion_gas.tex \
combustion_intro.tex \
combustion_thermo.tex \
comprs.tex \
cs_boundary_conditions.tex \
cs_solve_equation.tex \
elbase.tex \
ext_neighborhood.tex \
goveqn.tex \
gradrc.tex \
ground_water.tex \
cs_mass_flux.tex \
introd.tex \
itrmas.tex \
lagrangian.tex \
matrix.tex \
mesh_algo.tex \
mesh_quality.tex \
cs_solve_navier_stokes.tex \
operat.tex \
cs_velocity_prediction.tex \
cs_pressure_correction.tex \
spadis.tex \
timstp.tex \
turbul.tex \
summary.tex \
cs_turbulence_rij.tex \
cs_face_viscosity.tex \
cs_face_orthotropic_viscosity_vector.tex \
visecv.tex


TEXINPUTFILES =

# Distributed files

EXTRA_DIST = $(TEXMAIN) $(TEXOTHERS) $(TEXINPUTFILES) graphics

# Generated pdf document

PDFMAIN = $(TEXMAIN:.tex=.pdf)

if HAVE_LATEX

# Environment variables for LaTeX

TEXINPUTS = $(top_srcdir)/docs/style/$(cs_tex_path_sep)$(top_builddir)/docs/style/$(cs_tex_path_sep)$(srcdir)//$(cs_tex_path_sep)$(builddir)//$(cs_tex_path_end)
BIBINPUTS = $(top_srcdir)/docs/style/
BSTINPUTS = $(top_srcdir)/docs/style/

# LaTeX commands

PDFLATEXCMD = export TEXINPUTS="$(TEXINPUTS)"; "$(PDFLATEX)"
BIBTEXCMD = export BIBINPUTS="$(BIBINPUTS)"; "$(BIBTEX)"
MAKEINDEXCMD = "$(MAKEINDEX)"

# generated index files

BIBMAIN = $(TEXMAIN:.tex=.aux)
GLOSSMAIN = $(TEXMAIN:.tex=.nlo)
INDEXMAIN = $(TEXMAIN:.tex=.nls)

# One redefines here the standard pdf targets of automake
# so that they handle generation from LaTeX documents.

pdf-local:
	@$(PDFLATEXCMD) $(TEXMAIN)
	@$(BIBTEXCMD) $(BIBMAIN)
	@$(PDFLATEXCMD) $(TEXMAIN)
	@$(MAKEINDEXCMD) $(GLOSSMAIN) -s nomencl.ist -o $(INDEXMAIN)
	@$(PDFLATEXCMD) $(TEXMAIN)

clean-local:
	-rm -f *.out *.log *.aux *.toc *.lof *.lot \
	 *.ilg *.idx *.bbl *.blg *.ind *.sym *.nsy \
	 *.nlo *.nls

install-pdf-local:
	$(MKDIR_P) "$(DESTDIR)$(pdfdir)"
	$(INSTALL_DATA) $(PDFMAIN) $(DESTDIR)$(pdfdir)

distclean-local: clean-local
	-rm -f $(PDFMAIN)

# Generate the documentation at dist stage so that the user do not need
# to have LaTeX (and mandatory tools).

dist-hook: pdf
	$(INSTALL_DATA) $(PDFMAIN) $(distdir)

endif

# Install the documentation if available

install-data-local:
	if test -f $(srcdir)/$(PDFMAIN); then \
	  $(MKDIR_P) "$(DESTDIR)$(pdfdir)"; \
	  $(INSTALL_DATA) $(srcdir)/$(PDFMAIN) $(DESTDIR)$(pdfdir); \
	fi
