# -*- coding: utf-8 -*-

#-------------------------------------------------------------------------------

# This file is part of code_saturne, a general-purpose CFD tool.
#
# Copyright (C) 1998-2024 EDF S.A.
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

#-------------------------------------------------------------------------------
# Standard modules import
#-------------------------------------------------------------------------------

import os, sys
import shutil, re
import subprocess
import threading
import string
import time
import logging
import fnmatch
import math
import configparser
from collections import OrderedDict

#-------------------------------------------------------------------------------
# Application modules import
#-------------------------------------------------------------------------------

from code_saturne.base.cs_compile import files_to_compile, compile_and_link
from code_saturne.base import cs_create, cs_batch, cs_xml_reader
from code_saturne.base.cs_case import case_state, get_case_state
from code_saturne.base.cs_create import set_executable, create_local_launcher
from code_saturne.base import cs_exec_environment, cs_run_conf

from code_saturne.model import XMLengine
from code_saturne.studymanager.cs_studymanager_pathes_model import PathesModel

from code_saturne.studymanager.cs_studymanager_parser import Parser
from code_saturne.studymanager.cs_studymanager_texmaker import Report

try:
    from code_saturne.studymanager.cs_studymanager_drawing import Plotter
except Exception:
    print("Warning: import studymanager Plotter failed. Plotting disabled.\n")
    pass

from code_saturne.studymanager.cs_studymanager_run import run_studymanager_command
from code_saturne.studymanager.cs_studymanager_xml_init import smgr_xml_init

#-------------------------------------------------------------------------------
# log config.
#-------------------------------------------------------------------------------

logging.basicConfig()
log = logging.getLogger(__file__)
#log.setLevel(logging.DEBUG)
log.setLevel(logging.NOTSET)

#-------------------------------------------------------------------------------

def nodot(item):
    return item[0] != '.'

#-------------------------------------------------------------------------------

def create_base_xml_file(filepath, pkg):
    """Create studymanager XML file.
    """

    filename = os.path.basename(filepath)
    msg = None
    if os.path.isfile(filepath):
        msg = "Can not create XML file of parameter:\n" \
            + filepath + " already exists."
        return None, msg

    # using xml engine from code_saturne GUI
    smgr = XMLengine.Case(package=pkg, studymanager=True)
    smgr['xmlfile'] = filename
    pm = PathesModel(smgr)

    # empty repo and dest
    pm.setRepositoryPath('')
    pm.setDestinationPath('')

    return smgr, msg

#-------------------------------------------------------------------------------

def init_xml_file_with_study(smgr, studyp, pkg):
    """Initialize XML file with study content and save it.
    """

    smgr_node = smgr.xmlGetNode('studymanager')

    studyd = os.path.basename(studyp)
    st_node = smgr_node.xmlInitChildNode('study', label = studyd)
    st_node['status'] = "on"

    cases = []
    for elt in os.listdir(studyp):
        eltd = os.path.join(studyp, elt)
        if os.path.isdir(eltd):
            if isCase(eltd, pkg):
                cases.append(elt)

    cases.sort()
    for case in cases:
        c_node = st_node.xmlInitChildNode("case", label = case)
        c_node['status']  = "on"

    smgr.xmlSaveDocument()

#-------------------------------------------------------------------------------

def isStudy(dirpath, pkg):
    """Try to determine if dirpath is a code_saturne study directory.
       True if a studymanager xml file is found
    """

    is_study = False
    if os.path.isdir(dirpath):
        for elt in os.listdir(dirpath):
            if ".xml" in str(elt):
                xml_file = os.path.join(dirpath, elt)
                if os.path.isfile(xml_file):
                    try:
                        smgr = XMLengine.Case(package=pkg, file_name=xml_file,
                                              studymanager=True)
                        Parser(xml_file, doc=smgr.doc)
                        is_study = True
                    except:
                        pass

    return is_study

#-------------------------------------------------------------------------------

def isCase(dirpath, pkg):
    """Try to determine if dirpath is a code_saturne case directory.
       True if a code_saturne xml file is found
    """

    # Verify that DATA folder exists with a xml file inside
    datad = os.path.join(dirpath, 'DATA')

    is_case = False
    if os.path.isdir(datad):
        for elt in os.listdir(datad):
            if ".xml" in str(elt):
                xml_file = os.path.join(datad, elt)
                if os.path.isfile(xml_file):
                    try:
                        setup = XMLengine.Case(package=pkg, file_name=xml_file)
                        cs_xml_reader.Parser(fileName = xml_file)
                        is_case = True
                    except:
                        pass

    return is_case

#===============================================================================
# Case class
#===============================================================================

class Case(object):
    def __init__(self,
                 pkg,
                 rlog,
                 diff,
                 parser,
                 study_label,
                 study_index,
                 smgr_cmd,
                 data,
                 repo,
                 dest,
                 resource_config=None):
        """
        @type data: C{Dictionary}
        @param data: contains all keyword and value read in the parameters file
        """
        self.__log_file    = rlog
        self.__diff        = diff
        self.__parser      = parser
        self.__data        = data
        self.__repo        = repo
        self.__dest        = dest

        self.pkg           = pkg
        self.study         = study_label
        self.study_index   = study_index

        # Driven by smgr xml file
        self.node          = data['node']
        self.label         = data['label']
        self.run_id        = data['run_id']
        self.n_procs       = data['n_procs']
        self.depends       = data['depends']
        self.expected_time = data['expected_time']
        self.parametric    = data['parametric']
        self.notebook      = data['notebook']
        self.kw_args       = data['kw_args']

        # Driven by smgr command line options but can be modified later
        self.compute       = smgr_cmd['compute']
        self.compare       = smgr_cmd['compare']
        self.plot          = smgr_cmd['post']
        self.sheet         = smgr_cmd['sheet']
        self.state         = smgr_cmd['state']

        self.is_compiled   = "not done"
        self.is_time       = None
        self.is_plot       = "not done"
        self.no_restart    = False
        self.threshold     = "default"
        self.diff_value    = [] # list of differences (in case of comparison)
        self.m_size_eq     = True # mesh sizes equal (in case of comparison)
        self.subdomains    = None
        self.level         = None # level of the node in the dependency graph
        self.job_id        = None

        # Run_dir and Title are based on study, label and run_id
        self.resu = "RESU"
        self.run_dir = os.path.join(self.__dest, self.label, self.resu,
                                    self.run_id)
        self.title = self.study + "/" + self.label + "/" +  self.resu + "/" \
                   + self.run_id

        # Check for coupling
        # TODO: use run.cfg info, so as to allow another coupling parameters
        #       path ('run.cfg' is fixed, 'coupling_parameters.py' is not).

        coupling = False
        run_conf = None
        run_config_path = os.path.join(self.__repo, self.label, "run.cfg")
        if os.path.isfile(run_config_path):
            run_conf = cs_run_conf.run_conf(run_config_path, package=self.pkg)
            if run_conf.get("setup", "coupled_domains") != None:
                coupling = True

        if coupling:
            self.resu = "RESU_COUPLING"

            # Apply coupling parameters information

            from code_saturne.base import cs_case_coupling

            coupled_domains = run_conf.get_coupling_parameters()

            self.subdomains = []
            for d in coupled_domains:
                if d['solver'].lower() in ('code_saturne', 'neptune_cfd'):
                    self.subdomains.append(d['domain'])

            self.run_dir = os.path.join(self.__dest, self.label, self.resu,
                                        self.run_id)

            self.title = self.study + "/" + self.label + "/" +  self.resu \
                       + "/" + self.run_id

        self.exe = os.path.join(pkg.get_dir('bindir'),
                                pkg.name + pkg.config.shext)

        # Query number of processes and expected computation time

        if not self.n_procs or not self.expected_time:
            if run_conf == None:
                run_config_path = os.path.join(self.__repo, self.label,
                                               "DATA", "run.cfg")
                if os.path.isfile(run_config_path):
                    run_conf = cs_run_conf.run_conf(run_config_path,
                                                    package=self.pkg)

            if resource_config:
                config = self.__query_config__(run_conf, resource_config)

                # Information in data (i.e. smgr xml file) dominates over config
                if not self.n_procs:
                    self.n_procs = config[0]
                if not self.expected_time:
                    self.expected_time = config[1]

        # Check for dependency
        # Dependancy given in smgr xml file overwrites parametric arguments
        # or dependency defined in setup.xml
        if not self.depends:
            self.depends = self.__query_dependence__()

    #---------------------------------------------------------------------------

    def __query_config__(self, run_conf, resource_config):
        """
        Determine number of processes required and expected time for a given
        case based on its run.cfg and install configurations.
        """

        n_procs = None
        expected_time = None

        tags = []
        if self.__data['tags']:
            tags = self.__data['tags']

        if run_conf == None:
            return [int(resource_config['resource_n_procs']),
                    int(resource_config['resource_exp_time'])]

        specific_resource_name_rid = ""
        specific_resource_name_tag = []
        resource_name = resource_config['resource_name']

        # Try to find specific configuration with run_id in the form
        # [<resource>/run_id=<run_id>] otherwise use classical resource
        # also try to find specific configuration with tag in the form
        # [<resource>/tag=<run_id>] otherwise use classical resource

        if resource_name:
            resource_name = resource_name.lower()
            specific_resource_name_rid = resource_name + "/run_id=" + self.run_id.lower()
            for tag in tags:
                specific_resource_name_tag.append(resource_name + "/tag=" + tag.lower())
            common_tag = [tmp for tmp in specific_resource_name_tag if tmp in run_conf.sections]

        if not resource_name \
           or (not resource_name in run_conf.sections and not common_tag
               and not specific_resource_name_rid in run_conf.sections):
            resource_name = resource_config['batch_name']
            if resource_name:
                specific_resource_name_rid = resource_name + "/run_id=" + self.run_id.lower()
                specific_resource_name_tag = []
                for tag in tags:
                    specific_resource_name_tag.append(resource_name + "/tag=" + tag.lower())
            common_tag = [tmp for tmp in specific_resource_name_tag if tmp in run_conf.sections]
            if not resource_name \
               or (not resource_name in run_conf.sections and not common_tag
                   and not specific_resource_name_rid in run_conf.sections):
                resource_name = 'job_defaults'

        specific_resource_name_rid = resource_name + "/run_id=" + self.run_id.lower()
        specific_resource_name_tag = []
        for tag in tags:
            specific_resource_name_tag.append(resource_name + "/tag=" + tag.lower())

        # specific resource defined with run_id dominates over tag
        if specific_resource_name_rid in run_conf.sections:
            resource_name = specific_resource_name_rid
        else:
            common_tag = [tmp for tmp in specific_resource_name_tag if tmp in run_conf.sections]
            if common_tag:
                resource_name = str(common_tag[0])

        n_procs = run_conf.get(resource_name, 'n_procs')
        if n_procs == None:
            self.job_header_lines = None
            if resource_config['batch']:
                job_header = run_conf.get(resource_name, 'job_header')
                if job_header != None:
                    job_header_lines = job_header.split(os.linesep)
                    if job_header_lines != None:
                        batch = cs_batch.batch(self.pkg)
                        n_procs = batch.get_n_procs(job_header_lines)

        if n_procs == None:
            n_procs = run_conf.get('job_defaults', 'n_procs')

        if not n_procs:
            n_procs = resource_config['resource_n_procs']

        expected_time = run_conf.get(resource_name, 'expected_time')
        if expected_time:
            # Convert expected time in minutes
            tmp = str(expected_time)
            ind = tmp.find(":")
            expected_time = int(tmp[:ind]) * 60 + int(tmp[-2:])
        else:
            expected_time = resource_config['resource_exp_time']

        n_procs = int(n_procs)
        expected_time = int(expected_time)

        return [n_procs, expected_time]

    #---------------------------------------------------------------------------

    def __query_dependence__(self):
        """
        Check for dependency in DATA/setup.xml overwriten by dependency in
        parametric arguments
        Remark: we only take into account dependencies to a non existing run
        folder as it is used to manage slurm submissions
        """

        depends = None

        # Check dependency in DATA/setup.xml
        data_file = ""
        data_folder = os.path.join(self.__repo, self.label, "DATA")
        if os.path.isdir(data_folder):
            dir_list = os.listdir(data_folder)
            if "setup.xml" in dir_list:
                data_file = os.path.join(self.__repo, self.label, "DATA",
                                         "setup.xml")
        # Read setup.xml
        # Format <restart path="../CASE/RESU/run_id/checkpoint"/>
        path = None
        if os.path.isfile(data_file):
            data = cs_xml_reader.Parser(fileName = data_file)
            calc_node = cs_xml_reader.getChildNode(data.root,
                                                   'calculation_management')
            if calc_node != None:
                sr_node = cs_xml_reader.getChildNode(calc_node, 'start_restart')
                if sr_node != None:
                   node = cs_xml_reader.getChildNode(sr_node, 'restart')
                   if node != None:
                       path = str(node.getAttribute('path'))

        # Convert path in smgr dependency (STUDY/CASE/run_id)
        if path:
            list_path = path.split(os.sep)
            if len(list_path) > 4:
                run_id_depends = list_path[-2]
                case_depends = list_path[-4]
                depends = os.path.join(self.study, case_depends,
                                       list_path[-3], run_id_depends)

        # check dependency in parametric parameters
        # it overwrite data from setup.xml
        if self.parametric:
            list_param = self.parametric.split()
            for tag in ["-r", "--restart"]:
                if tag in list_param:
                    index = list_param.index(tag)
                    run_id_depends = list_param[index+1]
                    run_path = os.path.join(self.study, self.label,
                                            self.resu, run_id_depends)
                    # run_id_depends could contain path to another case
                    depends = os.path.normpath(run_path)

        # check the existence of the dependency run folder
        # remove dependency in case of sucessfull run
        # disable run in case of problem
        if depends:
            run_tmp = os.path.join(self.__dest, ".." , depends)
            run_folder = os.path.normpath(run_tmp)
            if os.path.isdir(run_folder):
                is_coupling = self.subdomains != None
                state, info = get_case_state(run_folder,
                                             coupling=is_coupling,
                                             run_timeout=3600)

                if state == case_state.FINALIZED:
                    # no dependency as run already finished
                    return None
                else:
                    # restart cannot be used
                    self.no_restart = True
                    self.compute = False
                    self.compare = False
                    self.plot = False
                    return depends
            else:
                # keep dependency
                return depends
        else:
            return None

    #---------------------------------------------------------------------------

    def prepare_run_folder(self):
        """
        Prepare a run folder in destination directory run_dir
        """
        log_lines = []
        e = os.path.join(self.pkg.get_dir('bindir'), self.exe)
        home = os.getcwd()
        os.chdir(self.__dest)

        have_status_prepared = False

        # Create log file in dest (will be moved later and renamed to
        # run_case.log, but named with run_id here in case of asynchronous
        # preparation of multiple runs).

        log_path = os.path.join(self.__dest, "run_" + self.label
                                + "_" + self.run_id + ".log")

        log_run = open(log_path, mode='w')

        if self.subdomains:
            if not os.path.isdir(self.label):
                os.mkdir(self.label)
            os.chdir(self.label)
            refdir = os.path.join(self.__repo, self.label)
            retval = 1
            resu_coupling = None
            for node in os.listdir(refdir):
                ref = os.path.join(self.__repo, self.label, node)

                # only loop on code_saturne subdomains
                if node in self.subdomains:

                    # generate folder in dest/STUDY/CASE/RESU_COUPLING/
                    cmd = e + " run --stage --case " + refdir \
                            + " --dest " + self.__dest \
                            + " --id " + self.run_id

                    if self.notebook:
                        cmd += " --notebook-args " + self.notebook

                    if self.parametric:
                        cmd += " --parametric-args " + '"' + self.parametric + '"'

                    if self.kw_args:
                        if self.kw_args.find(" ") < 0:
                            self.kw_args += " "  # workaround for arg-parser issue
                        cmd += " --kw-args " + '"' + self.kw_args + '"'

                    node_retval, t = run_studymanager_command(cmd, log_run)

                    # negative retcode is kept
                    retval = min(node_retval,retval)

                elif not os.path.isdir(ref):
                    shutil.copy2(ref, node)

            create_local_launcher(self.pkg, self.__dest)
            os.chdir(self.__dest)
        else:
            refdir = os.path.join(self.__repo, self.label)

            cmd = e + " run --stage --case " + refdir \
                    + " --dest " + self.__dest \
                    + " --id " + self.run_id

            if self.notebook:
                cmd += " --notebook-args " + self.notebook

            if self.parametric:
                cmd += " --parametric-args " + '"' + self.parametric + '"'

            if self.kw_args:
                if self.kw_args.find(" ") < 0:
                    self.kw_args += " "  # workaround for arg-parser issue
                cmd += " --kw-args " + '"' + self.kw_args + '"'

            # Check if case has already been prepared in dest/STUDY/CASE

            if os.path.isfile(os.path.join(self.run_dir, "run_status.prepared")):
                have_status_prepared = True

            retval, t = run_studymanager_command(cmd, log_run)

        if retval == 0:
            log_lines += ['      * prepare run folder: {0} --> OK ({1} s)'.format(self.title, str(t))]

            # move log file tog run_dir
            new_log_path = os.path.join(self.run_dir, "run_case.log")
            os.replace(log_path, new_log_path)

        else:
            have_case_log = False
            if os.path.isfile(os.path.join(self.run_dir, "run_case.log")):
                have_case_log = True

            fail_info = 'FAILED'
            if have_case_log:
                if have_status_prepared:
                    fail_info = 'SKIPPED (already prepared)'
                else:
                    fail_info = 'SKIPPED (already present)'
            log_lines += ['      * prepare run folder: {0} --> {1} ({2} s)'.format(self.title, fail_info, str(t))]

            # we set to off computation except in case of already prepared run
            if not have_status_prepared:
                log_lines += ['        - see ' + log_path]
                self.compute = False

        os.chdir(home)
        return log_lines

    #---------------------------------------------------------------------------

    def __update_setup(self, subdir):
        """
        Update setup file in the Repository.
        """
        # Load setup.xml file in order to update it
        # with the __backwardCompatibility method.

        from code_saturne.model.XMLengine import Case

        msg = None
        for fn in os.listdir(os.path.join(self.__repo, subdir, "DATA")):
            fp = os.path.join(self.__repo, subdir, "DATA", fn)
            if os.path.isfile(fp):
                try:
                    fd = os.open(fp , os.O_RDONLY)
                    f = os.fdopen(fd)
                    l = f.readline()
                    f.close()
                except Exception:
                    continue
                xml_type = None
                if l.startswith('''<?xml version="1.0" encoding="utf-8"?><Code_Saturne_GUI'''):
                    xml_type = 'code_saturne'
                elif l.startswith('''<?xml version="1.0" encoding="utf-8"?><NEPTUNE_CFD_GUI'''):
                    xml_type = 'neptune_cfd'
                else:
                    continue
                try:
                    case = Case(package = self.pkg, file_name = fp)
                except:
                    msg = "Parameters file reading error.\n" \
                        + "This file is not in accordance with XML specifications."
                    return msg

                case['xmlfile'] = fp
                case.xmlCleanAllBlank(case.xmlRootNode())

                if xml_type == 'code_saturne':
                    from code_saturne.model.XMLinitialize import XMLinit as cs_solver_xml_init
                    cs_solver_xml_init(case).initialize()
                elif xml_type == 'neptune_cfd':
                    try:
                        from code_saturne.model.XMLinitializeNeptune import XMLinitNeptune as nc_solver_xml_init
                        nc_solver_xml_init(case).initialize()
                    except ImportError:
                        # Avoid completely failing an update of cases with
                        # mixed solver types when neptune_cfd is not available
                        # (will fail if really trying to run those cases)
                        msg = "Failed updating a neptune_cfd XML file as " \
                            + "neptune_cfd is not available."
                        pass

                case.xmlSaveDocument()

        return msg

    #---------------------------------------------------------------------------

    def update(self):
        """
        Update path for the script in the Repository.
        """
        # Load the setup file in order to update it
        # with the __backwardCompatibility method.

        if self.subdomains:
            cdirs = []
            for d in self.subdomains:
                cdirs.append(os.path.join(self.label, d))
        else:
            cdirs = (self.label,)

        error = None
        for d in cdirs:
            error = self.__update_setup(d)

        return error

    #---------------------------------------------------------------------------

    def test_compilation(self, study_path, log):
        """
        Test compilation of sources for current case (if some exist).
        @rtype: C{String}
        @return: compilation test status (None if no files to compile).
        """

        if self.subdomains:
            sdirs = []
            for sd in self.subdomains:
                sdirs.append(os.path.join(study_path, self.label, sd, 'SRC'))
        else:
            sdirs = (os.path.join(study_path, self.label, 'SRC'),)

        # compilation test mode
        dest_dir = None

        self.is_compiled = None
        retcode = 0

        # loop over subdomains
        for s in sdirs:
            src_files = files_to_compile(s)

            if len(src_files) > 0:
                self.is_compiled = "OK"
                retcode += compile_and_link(self.pkg, self.pkg.solver, s, dest_dir,
                                            stdout=log, stderr=log)

        if retcode > 0:
            self.is_compiled = "KO"

        return self.is_compiled

    #---------------------------------------------------------------------------

    def build_run_cmd(self, resource_name=None, mem_log=False):
        """
        Define run command with specified options.
        """

        e = os.path.join(self.pkg.get_dir('bindir'), self.exe)

        refdir = os.path.join(self.__repo, self.label)

        # After the stage within run_id folder in destination
        # do initialize, execute and finalize steps
        run_cmd = e + " run --no-stage" \
                + " --case " + refdir \
                + " --dest " + self.__dest \
                + " --id " + self.run_id

        if mem_log:
            run_cmd += " --mem-log"

        if self.kw_args:
            if self.kw_args.find(" ") < 0:
                self.kw_args += " "  # workaround for arg-parser issue
            run_cmd += " --kw-args " + '"' + self.kw_args + '"'

        n_procs = self.__data['n_procs']
        if n_procs:
            run_cmd += " -n " + n_procs

        if resource_name:
            run_cmd += " --with-resource " + resource_name

        return run_cmd

    #---------------------------------------------------------------------------

    def add_control_file(self, n_iter):
        """
        Add a control file in a run if needed.
        """

        if self.subdomains:
            path = os.path.join(self.run_dir,
                                self.subdomains[0],
                                'control_file')
        else:
            path = os.path.join(self.run_dir,
                                'control_file')

        # Create a control_file in run folder
        if not os.path.exists(path):
            control_file = open(path,'w')
            control_file.write("time_step_limit " + str(n_iter) + "\n")
            # Flush to ensure that control_file content is seen
            # when control_file is copied to the run directory on all systems
            # Is this useful ? Isn't flush automatic when closing ?
            control_file.flush()
            control_file.close

    #---------------------------------------------------------------------------

    def build_run_batch(self, resource_name=None, mem_log=False):
        """
        Launch run in RESU/run_id subdirectory in an batch mode.
        """
        run_cmd_batch = "cd " + self.run_dir + os.linesep

        run_cmd_batch += self.build_run_cmd(resource_name, mem_log)

        # append run_case.log in run_dir
        run_cmd_batch += " >> run_case.log 2>&1" + os.linesep + os.linesep

        return run_cmd_batch

    #---------------------------------------------------------------------------

    def run(self, resource_name=None, mem_log=False):
        """
        Launch run in RESU/run_id subdirectory.
        """
        home = os.getcwd()
        os.chdir(self.run_dir)

        run_cmd = self.build_run_cmd(resource_name, mem_log)

        # append run_case.log in run_dir
        file_name = os.path.join(self.run_dir, "run_case.log")
        log_run = open(file_name, mode='a')

        error, self.is_time = run_studymanager_command(run_cmd, log_run)

        mem_log_leak = False
        if mem_log:
            mem_log_leak_code = 2
            if error & mem_log_leak_code:
                mem_log_leak = True
                error -= mem_log_leak_code

        if error:
            self.compare = False
            self.plot = False

        os.chdir(home)

        return error, mem_log_leak

    #---------------------------------------------------------------------------

    def runCompare(self, studies, r, d, threshold, args, reference=None):
        home = os.getcwd()

        node = None

        if reference:
            result = os.path.join(reference, self.label, self.resu)
        else:
            result = os.path.join(self.__repo, self.label, self.resu)
        # check_dir called again here to get run_id (possibly date-hour)
        repo, msg = self.check_dir(node, result, r, "repo")
        if msg:
            studies.reporting(msg)
        repo = os.path.join(result, repo, 'checkpoint', 'main')
        if not os.path.isfile(repo):
            repo += '.csc'

        result = os.path.join(self.__dest, self.label, self.resu)
        # check_dir called again here to get run_id (possibly date-hour)
        dest, msg = self.check_dir(node, result, d, "dest")
        if msg:
            studies.reporting(msg)
        dest = os.path.join(result, dest, 'checkpoint', 'main.csc')

        cmd = self.__diff + ' ' + repo + ' ' + dest

        self.threshold = "default"
        if threshold != None:
            cmd += ' --threshold ' + threshold
            self.threshold = threshold

        if args != None:
            cmd += (" " + args)
            l = args.split()
            try:
                i = l.index('--threshold')
                self.threshold = l[i+1]
            except:
                pass

        l = subprocess.Popen(cmd,
                             shell=True,
                             executable=cs_exec_environment.get_shell_type(),
                             stdout=subprocess.PIPE,
                             universal_newlines=True).stdout
        lines = l.readlines()

        # list of field differences
        tab = []
        # meshes have same sizes
        m_size_eq = True

        # studymanager compare log only for field of real values
        for i in range(len(lines)):
            # only select line with "Type" (english and french) and ';'
            # since this should be only true for heads of section
            if lines[i].find("Type") != -1 and lines[i].find(";") != -1:
                line = [x.replace("\""," ").strip() for x in lines[i].split(";")]
                name = line[0]
                info = [x.split(":") for x in line[1:]]
                info = [[x[0].strip(),x[1].strip()] for x in info]

                # section with at least 2 informations (location, type) after
                # their name, and of type r (real)
                if len(info) >= 2 and info[1][1] in ['r4', 'r8']:
                    # if next line contains size, this means sizes are different
                    if lines[i+1].find("Taille") != -1 or lines[i+1].find("Size") != -1:
                        m_size_eq = False
                        break
                    else:
                        line = [x.strip() for x in lines[i+1].split(";")]
                        vals = [x.split(":") for x in line]
                        vals = [[x[0].strip(),x[1].strip()] for x in vals]
                        tab.append([name.replace("_", "\_"),
                                    vals[1][1],
                                    vals[2][1],
                                    self.threshold])

        os.chdir(home)

        return tab, m_size_eq

    #---------------------------------------------------------------------------

    def run_ok(self, run_dir):
        """
        Check if a result directory contains an error file
        or if it doesn't contain a summary file
        """
        if not os.path.isdir(run_dir):
            msg = "Error: the result directory %s does not exist." % run_dir
            return False, msg

        msg = ""
        ok = True

        f_error = os.path.join(run_dir, 'error')
        if os.path.isfile(f_error):
            ok = False
            msg += "the result directory %s in case %s " \
                   "contains an error file." % (os.path.basename(run_dir),
                                                self.title)

        f_summary = os.path.join(run_dir, 'summary')
        if not os.path.isfile(f_summary):
            ok = False
            msg += "the result directory %s in case %s " \
                   "does not contain any summary file." \
                   % (os.path.basename(run_dir), self.title)

        return ok, msg

    #---------------------------------------------------------------------------

    def check_dir(self, node, result, rep, attr):
        """
        Check coherency between xml file of parameters and repository or
        destination.
        """
        msg = "Warning: "

        if not os.path.isdir(result):
            msg += "the directory %s does not exist." %(result)
            return None, msg

        # 1. The result directory is given
        if rep != "":
            # check if it exists
            rep_f = os.path.join(result, rep)
            if not os.path.isdir(rep_f):
                msg += "the result directory %s does not exist." %(rep_f)
                return None, msg

            run_ok = self.run_ok(rep_f)
            if not run_ok[0]:
                return None, msg+run_ok[1]

        # 2. The result directory must be found/read automatically;
        elif rep == "":
            # check if there is at least one result directory.
            if len(list(filter(nodot, os.listdir(result)))) == 0:
                msg += "there is no result directory in %s." %(result)
                return None, msg

            # if no run_id is specified in the xml file
            # only one result directory allowed in RESU
            if len(list(filter(nodot, os.listdir(result)))) > 1 \
               and self.run_id == "run1":
                msg += "there are several result directories in %s " \
                       "and no run id specified." % (result)
                return None, msg

            rep = self.run_id
            # if no run_id is specified in the xml file
            # the only result directory present in RESU is taken
            if rep == "run1":
                rep = list(filter(nodot, os.listdir(result)))[0]

            rep_f = os.path.join(result, rep)
            if not os.path.isdir(rep_f):
                msg += "the result directory %s does not exist." %(rep_f)
                return None, msg

            run_ok = self.run_ok(rep_f)
            if not run_ok[0]:
                return None, msg+run_ok[1]

        return rep, None

    #---------------------------------------------------------------------------

    def get_state(self, run_timeout=3600):
        """
        Get state based on RESU/run_id subdirectory.
        """

        is_coupling = self.subdomains != None

        state, info = get_case_state(self.run_dir,
                                     coupling=is_coupling,
                                     run_timeout=run_timeout)

        return state, info

    #---------------------------------------------------------------------------

    def check_dirs(self, node, repo, dest, reference=None):
        """
        Check coherency between xml file of parameters and repository and destination.
        """
        msg = None

        if repo != None:
            # build path to RESU directory with path to study and case label in the repo
            if reference:
                result = os.path.join(reference, self.label, self.resu)
            else:
                result = os.path.join(self.__repo, self.label, self.resu)
            rep, msg = self.check_dir(node, result, repo, "repo")

        if dest != None:
            # build path to RESU directory with path to study and case label in the dest
            result = os.path.join(self.__dest, self.label, self.resu)
            rep, msg = self.check_dir(node, result, dest, "dest")

        return msg

    #---------------------------------------------------------------------------

    def check_file(self, folder, dest, file_name):
        """
        Verify the existence of file in destination
        """
        msg = None

        # build path to directory in destination
        file_addr = os.path.join(folder, dest, file_name)
        if not os.path.isfile(file_addr):
            msg = "    The file %s does not exist." %(file_addr)

        return msg

#===============================================================================
# Study class
#===============================================================================

class Study(object):
    """
    Create, run and compare all cases for a given study.
    """
    def __init__(self,
                 pkg,
                 parser,
                 study,
                 index,
                 exe,
                 dif,
                 smgr_cmd,
                 rlog,
                 n_procs=None,
                 with_tags=None,
                 without_tags=None,
                 resource_config=None):
        """
        Constructor.
          1. initialize attributes,
          2. build the list of the cases,
          3. build the list of the keywords from the case
        @param pkg: information on the solver executable: C{pkg}
        @param parser: instance of the parser: C{Parser}
        @param study: label of the current study: C{String}
        @param exe: name of the solver executable: C{code_saturne} or
        @    C{neptune_cfd}: C{String}
        @param dif: name of the diff executable: C{cs_io_dump -d}: C{String}
        @param smgr_cmd: smgr command option (compute, compare and post):
        @    C{list}
        @param rlog: log file: C{file}
        @param n_procs: number of requested processes: C{int}
        @param with_tags: list of tags given at the command line: C{List}
        @param without_tags: list of tags given at the command line: C{List}
        @param resource_config: name of the compute resource: C{String}
        """
        # Initialize attributes
        self.__package  = pkg
        self.__parser   = parser
        self.__main_exe = exe
        self.__diff     = dif
        self.__log_file = rlog

        # read repository and destination from smgr file
        # based on working directory if information is not available
        try:
            self.__repo = os.path.join(self.__parser.getRepository(),  study)
        except:
            self.__repo = os.path.join(os.getcwd(),  study)
        try:
            self.__dest = os.path.join(self.__parser.getDestination(), study)
        except:
            self.__dest = self.__repo

        if not os.path.isdir(self.__repo):
            print("Error: the directory %s does not exist" % self.__repo)
            sys.exit(1)

        self.label = study
        self.index = index

        self.cases = []
        self.matplotlib_figures = []
        self.input_figures = []
        self.case_labels = []

        # get list of cases in study
        on_cases = parser.getStatusOnCasesLabels(study)
        if not on_cases:

            print("\n\n\nWarning: no case defined in %s study\n\n\n" % study)

        else:

            for data in self.__parser.getStatusOnCasesKeywords(self.label):

                # n_procs given in smgr command line overwrites n_procs by case
                if n_procs:
                    data['n_procs'] = str(n_procs)

                # TODO: move tag's filtering to the graph level
                # not done for now as the POST step needs tags also
                # check if every tag passed by option --with-tags belongs to
                # list of tags of the current case
                tagged = False
                if with_tags and data['tags']:
                    tagged = all(tag in data['tags'] for tag in with_tags)
                elif not with_tags:
                    tagged = True

                # check if none of tags passed by option --without-tags
                # belong to list of tags of the current case
                exclude = False
                if without_tags and data['tags']:
                    exclude = any(tag in data['tags'] for tag in without_tags)

                # do not append case if tags do not match
                if tagged and not exclude:
                    c = Case(pkg,
                             self.__log_file,
                             self.__diff,
                             self.__parser,
                             self.label,
                             self.index,
                             smgr_cmd,
                             data,
                             self.__repo,
                             self.__dest,
                             resource_config=resource_config)
                    self.cases.append(c)
                    self.case_labels.append(c.label)

    #---------------------------------------------------------------------------

    def getRunDirectories(self):
        list_cases = []
        list_dir   = []

        for case in self.cases:
            list_cases.append(case.label)
            list_dir.append(case.run_dir)

        return " ".join(list_cases), " ".join(list_dir)

#===============================================================================
# Studies class
#===============================================================================

class Studies(object):
    """
    Manage all Studies and all Cases described in the files of parameters.
    """
    def __init__(self, pkg, options, exe, dif):
        """
        Constructor.
          1. create if necessary the destination directory,
          2. initialize the parser and the plotter,
          3. build the list of the studies,
          4. start the report.
        @type options: C{Structure}
        @param options: structure the parameters options.
        @type exe: C{String}
        @param exe: name of the solver executable: C{code_saturne} or C{neptune_cfd}.
        @type dif: C{String}
        @param dif: name of the diff executable: C{cs_io_dump -d}.
        """

        # try to determine if current directory is a study one
        cwd = os.getcwd()
        is_study = isStudy(cwd, pkg)
        studyd = None
        studyp = None
        if is_study or options.create_xml:
            # default study directory is current one
            studyp = cwd

        smgr = None
        self.__log = None
        self.__log_compile = None
        self.__log_file = None

        # Store options

        self.__pkg               = pkg
        self.__exe               = exe
        self.__filename          = options.filename
        self.__create_xml        = options.create_xml
        self.__update_smgr       = options.update_smgr
        self.__update_setup      = options.update_setup
        self.__force_rm          = options.remove_existing
        self.__disable_ow        = options.disable_overwrite
        self.__debug             = options.debug
        self.__state             = options.casestate
        self.__n_procs           = options.n_procs
        self.__filter_level      = options.filter_level
        self.__filter_n_procs    = options.filter_n_procs
        # Use the provided resoure if forced
        self.__resource_name     = options.resource_name
        self.__quiet             = options.quiet
        self.__n_iter            = options.n_iterations
        self.__compare           = options.compare
        self.__ref               = options.reference
        self.__postpro           = options.post
        self.__slurm_batch       = options.slurm_batch
        self.__slurm_batch_size  = options.slurm_batch_size
        # Convert in minutes
        self.__slurm_batch_wtime = options.slurm_batch_wtime * 60
        self.__slurm_batch_args  = options.slurm_batch_args
        self.__sheet             = options.sheet
        self.__default_fmt       = options.default_fmt
        # do not use tex in matpl(built-in mathtext is used instead)
        self.__dis_tex           = options.disable_tex
        # tex reports compilationpdflatex
        self.__pdflatex          = not options.disable_pdflatex
        # use the mem-log option for unfreed pointers
        self.__mem_log           = options.mem_log

        # Query install configuration and current environment
        # (add number of procs based on resources to install
        # configuration).

        resource_config = cs_run_conf.get_install_config_info(pkg)
        if self.__resource_name:
            resource_config['resource_name'] = self.__resource_name
        self.__resource_name = resource_config['resource_name']

        exec_env = cs_exec_environment.exec_environment(pkg,
                                                        wdir=None,
                                                        n_procs=options.n_procs,
                                                        n_procs_default=None,
                                                        n_threads=None)

        resource_config['resource_n_procs'] = exec_env.resources.n_procs
        if not resource_config['resource_n_procs']:
            resource_config['resource_n_procs'] = 1

        # Store smgr command options
        smgr_cmd = {"compute" : options.runcase,
                    "compare" : options.compare,
                    "post"    : options.post,
                    "sheet"   : options.sheet,
                    "state"   : options.casestate}

        # Default expected time in minutes
        resource_config['resource_exp_time'] = 180

        # Default batch size and wall time for SLURM batch mode
        if self.__slurm_batch:
            if not self.__slurm_batch_size:
                self.__slurm_batch_size = 1
            if not self.__slurm_batch_wtime:
                self.__slurm_batch_wtime = 8 * 60  # 8 hours in minutes

        rm_template = resource_config['batch']
        if rm_template:
            rm_template = os.path.basename(rm_template).lower()
            i = rm_template.rfind(".")
            if i > -1:
                rm_template = rm_template[i+1:]
        resource_config['batch_name'] = rm_template

        # Create file of parameters

        filename = options.filename
        if self.__create_xml:
            if filename is None:
                studyd = os.path.basename(studyp)
                filename = "smgr.xml"
                options.filename = filename

            filepath = os.path.join(studyp, filename)
            smgr, error = create_base_xml_file(filepath, pkg)

            if error:
                msg = "Error: can not create smgr xml file\n"
                self.reporting(msg, report=False, exit=False)
                self.reporting(error, stdout=False, report=False, exit=True)
            else:
                init_xml_file_with_study(smgr, studyp, pkg)
                self.reporting(" ", report=False)
                self.reporting(" New smgr.xml file was created succesfully",
                               report=False)
                self.reporting(" ", report=False)
                return

        if filename is None:
            msg = "Error: a file of parameters must be specified or created" \
                + " for studymanager to run.\n" \
                + "See help message and use '--file' or '--create-xml'" \
                + " option."
            self.reporting(msg, stdout=False, report=False, exit=True)

        # create a first smgr parser only for
        #   the repository verification and
        #   the destination creation

        if not os.path.isfile(filename):
            msg = "Error: specified XML parameter file for studymanager does" \
                + " not exist.\n" \
                + "File name should be smgr.xml or speficied with '-f' option"
            self.reporting(msg, stdout=False, report=False, exit=True)

        # call smgr xml backward compatibility

        if not smgr:
            try:
                smgr = XMLengine.Case(package=self.__pkg, file_name=filename,
                                      studymanager=True)
            except Exception as error:
                print("\n  /!\ ERROR while reading studymanager file :\n"
                      + str(error))
                sys.exit(error)

            smgr['xmlfile'] = filename

            # minimal modification of xml for now
            smgr_xml_init(smgr).initialize()
            if self.__update_smgr:
                smgr.xmlSaveDocument(prettyString=False,saveLink=True)

                self.reporting(" ", report=False)
                self.reporting(" SMGR parameter file was updated succesfully",
                               report=False)
                self.reporting(" Note that update is not necessary before the"\
                               + " run step", report=False)
                self.reporting(" ", report=False)

        self.__parser = Parser(filename, doc=smgr.doc)

        # set repository
        if len(options.repo_path) > 0:
            self.__parser.setRepository(options.repo_path)
        self.__repo = self.__parser.getRepository()
        if self.__repo:
            if not os.path.isdir(self.__repo):
                msg = "Error: repository path is not valid: " + self.__repo
                self.reporting(msg, stdout=False, report=False, exit=True)
        else: # default value
            # if current directory is a study
            # set repository as directory containing the study
            if is_study:
                studyd = os.path.basename(studyp)
                self.__parser.setRepository(os.path.join(studyp,".."))
                self.__repo = self.__parser.getRepository()

                # check consistency of the study name
                label = self.__parser.getStudiesLabel()
                if (len(label) == 1) and (studyd != label[0]):
                    msg = "The name of the current repository directory " \
                        + "differs from the name in the smgr xml file.\n" \
                        + "Please check consistency.\n"
                    self.reporting(msg, stdout=False, report=False, exit=True)

            else:
                msg = "Can not set a default repository directory:\n" \
                    + "current directory is apparently not a study (no smgr" \
                    + " xml file).\n" \
                    + "Add a repository path to the parameter file or use" \
                    + " the command line option (--repo=..)."
                self.reporting(msg, stdout=False, report=False, exit=True)

        # set destination
        self.__dest = None
        if len(options.dest_path) > 0:
            self.__parser.setDestination(options.dest_path)
        self.__dest = self.__parser.getDestination()
        if not self.__dest: # default value
            # if current directory is a study
            # set destination as a directory "../RUN_(study_name)
            if is_study and studyd != None:
                self.__parser.setDestination(os.path.join(studyp,
                                                          "../RUN_"+studyd))
                self.__dest = self.__parser.getDestination()
            else:
                msg = "Can not set a default destination directory:\n" \
                    + "current directory is apparently not a study (no smgr" \
                    + " xml file).\n" \
                    + "Add a destination path to the parameter file or use" \
                    + " the command line option (--dest=..).\n"
                self.reporting(msg, stdout=False, report=False, exit=True)

        if options.runcase or options.compare or options.post or options.sheet:

            # create if necessary the destination directory

            if not os.path.isdir(self.__dest):
                os.makedirs(self.__dest)

            # copy the updated smgr file in destination for restart

            dest_filename = os.path.join(self.__dest, os.path.basename(filename))
            try:
                smgr['xmlfile'] = dest_filename
                smgr.xmlSaveDocument(prettyString=False)
            except:
                pass

            # create definitive parser for smgr file in destination

            self.__parser = Parser(dest_filename)
            self.__parser.setDestination(self.__dest)
            self.__parser.setRepository(self.__repo)

            if options.debug:
                msg = " Studies repository: " + self.__repo + "\n" \
                  + " Studies destination: " + self.__dest
                self.reporting(msg, report=False)

            # create studymanager log file

            self.__log_name = os.path.join(self.__dest, options.log_file)
            self.__log_file = open(self.__log_name, "w")

            # create plotter and post log file

            if options.post:
                try:
                    self.__plotter = Plotter(self.__parser)
                except Exception:
                    self.__plotter = None
                self.__log_post_name = os.path.join(self.__dest,
                                                    "smgr_post_pro.log")
                self.__log_post_file = open(self.__log_post_name, "w")

        # create list of restricting and excluding tags

        self.__with_tags = None
        if options.with_tags:
            with_tags = re.split(',', options.with_tags)
            self.__with_tags = [tag.strip() for tag in with_tags]
        self.__without_tags = None
        if options.without_tags:
            without_tags = re.split(',', options.without_tags)
            self.__without_tags = [tag.strip() for tag in without_tags]

        # build the list of the studies

        self.labels  = self.__parser.getStudiesLabel()
        self.n_study = len(self.labels)
        self.studies = []
        index = -1
        for l in self.labels:
            index += 1
            self.studies.append( [l, Study(self.__pkg, \
                                           self.__parser, \
                                           l, \
                                           index, \
                                           exe, \
                                           dif, \
                                           smgr_cmd, \
                                           self.__log_file, \
                                           options.n_procs, \
                                           self.__with_tags, \
                                           self.__without_tags, \
                                           resource_config=resource_config)] )
            if options.debug:
                self.reporting(" Append study:" + l, report=False)

        # Handle relative paths:
        if self.__ref:
            if not os.path.isabs(self.__ref):
                self.__ref = os.path.join(os.getcwd(), self.__ref)

    #---------------------------------------------------------------------------

    def getDestination(self):
        """
        @rtype: C{String}
        @return: destination directory of all studies.
        """
        if self.__dest is None:
            msg = " cs_studymanager_study.py >> Studies.getDestination()" \
                + " >> self.__dest is not set"
            self.reporting(msg, exit=True)
        return self.__dest

    #---------------------------------------------------------------------------

    def getRepository(self):
        """
        @rtype: C{String}
        @return: repository directory of all studies.
        """
        if self.__repo is None:
            msg = " cs_studymanager_study.py >> Studies.getRepository()" \
                + " >> self.__repo is not set"
            self.reporting(msg, exit=True)
        return self.__repo

    #---------------------------------------------------------------------------

    def reporting(self, msg, stdout=True, report=True, status=False,
                  exit=False):
        """
        Write message on standard output and/or in report.
        @type l: C{String}
        @param l: the sentence to be written.
        """
        s  = ""
        if not status:
            s = chr(10)

        if stdout and not self.__quiet:
            sys.stdout.write (msg + chr(13) + s)
            sys.stdout.flush()

        if report:
            if self.__log_file is not None:
                self.__log_file.write(msg + '\n')
                self.__log_file.flush()

        if exit:
            sys.exit(msg)

    #---------------------------------------------------------------------------

    def report_action_location(self, header_msg, destination=True):
        """
        Add a message to report/stdout at head of sections
        specifying if action is performed in dest or repo.
        """
        if destination:
            header_msg = header_msg + " (in destination)"
        else:
            header_msg = header_msg + " (in repository)"

        self.reporting(header_msg)

    #---------------------------------------------------------------------------

    def updateSetup(self):
        """
        Update setup files in all cases.
        """

        for l, s in self.studies:
            self.reporting('  o In repository: ' + l, report=False)
            for case in s.cases:
                self.reporting('    - update setup file in %s' % case.label,
                               report=False)
                error = case.update()
                if error:
                   self.reporting(error, stdout=False, report=False, exit=True)

        self.reporting('',report=False)

    #---------------------------------------------------------------------------

    def create_studies(self, run_step):
        """
        Create all studies and all cases.
        """

        self.reporting("  o Create all studies and run folders")
        if self.__force_rm:
            self.reporting("    /!\ All earlier run folders are erased (option"
                           " --rm activated)")
        else:
            self.reporting("    /!\ All earlier run folders will not be erased."
                           " Use --rm option to do so.")

        study_list = []
        case_list = []

        for case in self.graph.graph_dict:

            # first step: create study of the case if necessary
            study = case.study
            if study not in study_list:
                self.create_study(study)
                study_list.append(study)

            # only run step requires to create cases
            if run_step:
                # second step: clean RESU folder if necessary
                case_name = case.study + "/" + case.label
                if case_name not in case_list and self.__force_rm:
                    case_list.append(case_name)
                    # Build short path to RESU dir. such as 'CASE1/RESU'
                    _dest_resu_dir = os.path.join(self.__dest, case.study,
                                                  case.label, case.resu)
                    if os.path.isdir(_dest_resu_dir):
                        if os.listdir(_dest_resu_dir):
                            shutil.rmtree(_dest_resu_dir)
                            os.makedirs(_dest_resu_dir)
                            self.reporting("  All earlier results in case %s/%s "
                                           "are removed (option --rm activated)"
                                           %(case.label, case.resu))

                # thrid step: prepare run folder
                log_lines = case.prepare_run_folder()

                for line in log_lines:
                    self.reporting(line)

        self.reporting('')

    #---------------------------------------------------------------------------

    def create_study(self, study):

        dest_study = os.path.join(self.__dest, study)
        repo_study = os.path.join(self.__repo, study)

        new_study = False
        home = os.getcwd()
        os.chdir(self.__dest)
        # Create study if necessary
        if not os.path.isdir(dest_study):
            new_study = True
            # build instance of study class
            cr_study = cs_create.study(self.__pkg, study)    # quiet

            create_msg = "    - Create study " + study
            self.report_action_location(create_msg)

            # create empty study
            cr_study.create()

        # Write content of POST and REPORT in destination
        # Content is overwritten by default if the study already exist
        # Option --dow can be used to disable overwriting
        # POST is not overwritten if --report option is used without --post
        if new_study or not self.__disable_ow:

            write_post = new_study or \
                         not (self.__sheet and not self.__postpro)

            # Copy external scripts for post-processing
            if write_post:
                if not new_study:
                    self.reporting("    /!\ POST folder is overwritten in %s"
                                   " use option --dow to disable overwrite" %study)
                ref = os.path.join(repo_study, "POST")
                if os.path.isdir(ref):
                    des = os.path.join(dest_study, "POST")
                    shutil.rmtree(des)
                    try:
                        shutil.copytree(ref, des)
                    except:
                        self.reporting("    /!\ ERROR while copying POST folder"
                                       " in %s" %study)
            else:
                if self.__sheet and not self.__postpro:
                    self.reporting("    /!\ POST folder is not overwritten in %s"
                                   " as --report option is used without --post"
                                   %study)

            # REPORT and STYLE folders are mandatory to generate
            # V&V description report
            # Copy REPORT folder
            ref = os.path.join(repo_study, "REPORT")
            if os.path.isdir(ref):
                des = os.path.join(dest_study, "REPORT")
                if os.path.isdir(des):
                    self.reporting("    /!\ REPORT folder is overwritten in %s"
                                   " use option --dow to disable overwrite"
                                   %study)
                    shutil.rmtree(des)
                try:
                    shutil.copytree(ref, des)
                    # if file is present, remove write-up.pdf in destination
                    writeup = os.path.join(des, "write-up.pdf")
                    if os.path.isfile(writeup):
                        os.remove(writeup)
                except:
                    self.reporting("    /!\ ERROR while copying REPORT folder"
                                   " in %s" %study)

            else:
                if self.__sheet:
                    self.reporting("    /!\ REPORT folder is mandatory in STUDY"
                                   " folder to generate description report of"
                                   " study %s" %study)

            # Copy STYLE folder
            ref = os.path.join(self.__repo, "STYLE")
            if os.path.isdir(ref):
                des = os.path.join(self.__dest, "STYLE")
                if os.path.isdir(des):
                    shutil.rmtree(des)
                try:
                    shutil.copytree(ref, des)
                except:
                    self.reporting("    /!\ ERROR while copying STYLE folder")
            else:
                if self.__sheet:
                    self.reporting("    /!\ STYLE folder is mandatory in"
                                   " REPOSITORY to generate description report"
                                   " of study %s" %study)

            # Copy README file(s) if it exists
            # If a README file exists, then only this file is considered
            readme_files = [f for f in os.listdir(repo_study) if f.startswith('README')]
            if 'README' in readme_files:
                # Keep only this file and ignore the other README* files
                readme = os.path.join(repo_study, "README")
                if os.path.isfile(readme):
                    dest_file = os.path.join(dest_study, "README")
                    if os.path.isfile(dest_file):
                        self.reporting(f"    /!\ README file is overwritten in {study}"
                                       " use option --dow to disable overwrite")
                    shutil.copyfile(readme, dest_file)

            else:
                for f in readme_files:
                    readme = os.path.join(repo_study, f)
                    if os.path.isfile(readme):
                        dest_file = os.path.join(dest_study, f)
                        if os.path.isfile(dest_file):
                            self.reporting(f"    /!\ {f} file is overwritten in {study}"
                                           " use option --dow to disable overwrite")
                        shutil.copyfile(readme, dest_file)

        os.chdir(home)

    #---------------------------------------------------------------------------

    def dump_graph(self):
        """
        Dump dependency graph based on all studies and all cases.
        Can be limited to a sub graph is filters an tags are given
        """

        filter_level   = self.__filter_level
        filter_n_procs = self.__filter_n_procs

        if filter_level or filter_n_procs:
            self.reporting("  o Dump dependency graph with option :")
            self.reporting("     - level=" + str(filter_level))
            self.reporting("     - n_procs=" + str(filter_n_procs))

        # create the global graph with all cases of all studies without filtering
        global_graph = dependency_graph()
        for l, s in self.studies:
            for case in s.cases:
                if case.compute or case.compare or case.plot or case.sheet or \
                   case.state:
                    msg = global_graph.add_node(case)
                    if msg:
                        self.reporting("Warning in global graph: " + msg)

                if case.no_restart:
                    self.reporting(" WARNING: " + case.title + " was cancelled.")
                    self.reporting(" The restart folder seems problematic. See " + case.depends)

        # extract the sub graph based on filters and tags
        if filter_level is not None or filter_n_procs is not None:

            sub_graph = dependency_graph()
            for node in global_graph.graph_dict:

                # check if the level of the case is the targeted one
                # only effective if filter_level is prescribed
                target_level = True
                if filter_level is not None:
                    target_level = node.level is int(filter_level)

                # check if the number of procs of the case is the targeted one
                # only effective if filter_n_procs is prescribed
                target_n_procs = True
                if filter_n_procs is not None:
                    target_n_procs = node.n_procs is int(filter_n_procs)

                if target_level and target_n_procs:
                    msg = sub_graph.add_node(node)
                    if msg:
                        self.reporting("Warning in filtered graph: " + msg)

            self.graph = sub_graph

        else:
            self.graph = global_graph

        self.reporting('')

    #---------------------------------------------------------------------------

    def test_compilation(self):
        """
        Compile sources of all runs with compute attribute at on.
        """

        # create compilation log file in repository
        log_comp_name = os.path.join(self.__repo, "smgr_compilation.log")
        log_comp_file = open(log_comp_name, "w")

        iko = 0
        for l, s in self.studies:
            self.reporting('  o Compile study: ' + l + ' (in repository)',
                           report=False)
            for case in s.cases:

                # build case dir. (in repo.)
                study_path = os.path.join(self.__repo, case.study)

                # test compilation
                is_compiled = case.test_compilation(study_path,
                                                    log_comp_file)

                # report
                if is_compiled == "OK":
                    self.reporting('    - compile %s --> OK' %case.title,
                                   report=False)
                elif is_compiled == "KO":
                    self.reporting('    - compile %s --> FAILED' %case.title,
                                   report=False)
                    iko+=1

        self.reporting('',report=False)

        log_comp_file.close()

        if iko:
            self.reporting('  Error: compilation failed for %s case(s).\n'
                           %iko, report=False, exit=False)
        self.reporting('  Compilation logs available in  %s \n'
                       %log_comp_name, report=False, exit=False)

    #---------------------------------------------------------------------------

    def run(self):
        """
        Update and run all cases.
        Warning, if the markup of the case is repeated in the xml file of parameters,
        the run of the case is also repeated.
        """

        self.reporting("  o Run all cases")

        for case in self.graph.graph_dict:

            if case.compute and case.is_compiled != "KO":

                if self.__n_iter is not None:
                    case.add_control_file(self.__n_iter)

                self.reporting('    - running %s ...' % case.title,
                               stdout=True, report=False, status=True)

                error, mem_log_leak = case.run(resource_name = self.__resource_name,
                                      mem_log = self.__mem_log)
                if case.is_time:
                    is_time = "%s s" % case.is_time
                else:
                    is_time = "existed already"

                if not error:
                    self.reporting('    - run %s --> OK (%s)' \
                                   % (case.title, is_time))

                else:
                    self.reporting('    - run %s --> FAILED (%s)' \
                                   % (case.title, is_time))
                    self.reporting('      * see run_case.log in ' + \
                                   case.run_dir)

                if self.__mem_log and mem_log_leak:
                    self.reporting('    - run %s --> Leaks in memory logs' \
                                   % (case.title))
                    self.reporting('      * see cs_mem.log(.N) in %s' \
                                   % (case.run_dir))
                self.__log_file.flush()

        self.reporting('')

    #---------------------------------------------------------------------------

    def check_slurm_batches(self):
        """
        Verify configuration in slurm batch mode.
        If the number of case per batch is 1 (value by default) and the total
        number of cases with expected time lower than 5 minutes is greater than
        50, the maximum number of case per batch is set to 50.
        """

        self.reporting("  o Check slurm batch configuration")

        nb_cases_short = 0

        if self.__slurm_batch_size == 1:

            # loop on level (0 means without dependency)
            for level in range(self.graph.max_level+1):
                # loop on number of processes
                for nproc in range(self.graph.max_proc):
                    # loop on cases of the sub graph
                    for case in self.graph.extract_sub_graph(level,nproc+1).graph_dict:

                        if case.expected_time < 5:
                            nb_cases_short += 1

            if nb_cases_short > 50:
                self.__slurm_batch_size = 50
                self.reporting("  WARNING: more than 50 cases with expected " +
                               "time lower than 5 minutes were detected\n" +
                               "           slurm_batch_size was set to 50\n")

        self.reporting('')

    #---------------------------------------------------------------------------

    def run_slurm_batches(self, state_file_name, job_name):
        """
        Run all cases in slurm batch mode.
        The number of case per batch is limited by a maximum number and a maximum
        wall time given by smgr options. Expected times are used to compute wall
        time.
        """

        slurm_batch_template = """#!/bin/sh
#SBATCH --ntasks={0}
#SBATCH --time={1}:{2}:00
#SBATCH --output=vnv_{3}
#SBATCH --error=vnv_{3}
#SBATCH --job-name={4}_{3}
"""
        cur_batch_id = 0
        tot_job_id_list = []

        # create folder in destination
        slurm_file_dir = os.path.join(self.__dest, "slurm_files")
        if not os.path.isdir(slurm_file_dir):
          os.makedirs(slurm_file_dir)
        os.chdir(slurm_file_dir)

        self.reporting("  o Run all cases in slurm batch mode")

        # loop on level (0 means without dependency)
        for level in range(self.graph.max_level+1):

            # loop on number of processes
            for nproc in range(self.graph.max_proc):

                batch_cmd = ""
                cur_batch_size = 0
                batch_total_time = 0

                # dependency job id list for the current batch
                dep_job_id_list = []
                # list of cases of the current batch
                cases_list = []

                # loop on cases of the sub graph
                for case in self.graph.extract_sub_graph(level,nproc+1).graph_dict:

                    if case.compute and case.is_compiled != "KO":

                        if self.__n_iter is not None:
                            case.add_control_file(self.__n_iter)

                        # check future submission total time and submit previous batch if not empty
                        submit_prev = False
                        if (cur_batch_size == 0) or (batch_total_time + \
                           float(case.expected_time) < self.__slurm_batch_wtime):
                            # append content of batch command with run of the case
                            batch_cmd += case.build_run_batch(mem_log=self.__mem_log)
                            cur_batch_size += 1
                            batch_total_time += float(case.expected_time)
                            cases_list.append(case)
                            if level > 0:
                                depend_id = self.graph.graph_dict[case].job_id
                                if depend_id not in dep_job_id_list:
                                    dep_job_id_list.append(depend_id)
                        else:
                            submit_prev = True

                        # submit once batch size or wall time is reached
                        if cur_batch_size >= self.__slurm_batch_size or submit_prev or \
                           batch_total_time >= self.__slurm_batch_wtime:
                            slurm_batch_name = "slurm_batch_file_" + \
                                               str(cur_batch_id) + ".sh"
                            slurm_batch_file = open(slurm_batch_name, mode='w')

                            # fill file with template
                            hh, mm = divmod(batch_total_time, 60)
                            cmd = slurm_batch_template.format(nproc+1,
                                                              math.ceil(hh),
                                                              math.ceil(mm),
                                                              cur_batch_id,
                                                              job_name)

                            # add exclusive option to batch template for
                            # computation with at least 6 processes
                            # force same processor otherwise
                            if nproc+1 > 5:
                                cmd += "#SBATCH --exclusive\n"
                            else:
                                cmd += "#SBATCH --nodes=1\n"
                                cmd += "#SBATCH --ntasks-per-core=1\n"

                            # add user defined options if needed
                            if self.__slurm_batch_args:
                                for _p in self.__slurm_batch_args:
                                    cmd += "#SBATCH " + _p + "\n"

                            cmd += "\n"
                            slurm_batch_file.write(cmd)

                            # fill file with batch command for several cases
                            slurm_batch_file.write(batch_cmd)
                            slurm_batch_file.flush()

                            # submit batch
                            if level < 1:
                                output = subprocess.check_output(['sbatch',
                                         slurm_batch_name])
                            else:
                                # list of dependency id should be in the
                                # :id1:id2:id3 format
                                list_id = ""
                                for item in dep_job_id_list:
                                    list_id += ":" + str(item)
                                if len(list_id) > 0:
                                    output = subprocess.check_output(['sbatch',
                                             "--dependency=afterany"
                                             + list_id, slurm_batch_name])
                                else:
                                    # empty list can occur with existing runs in study
                                    output = subprocess.check_output(['sbatch',
                                             slurm_batch_name])

                            # find job id with regex and store it
                            msg = output.decode('utf-8').strip()
                            match = re.search('\d{8}', msg)
                            job_id = match.group()
                            tot_job_id_list.append(job_id)
                            self.reporting('    - %s ...' % msg)
                            for item in cases_list:
                                item.job_id = job_id

                            batch_cmd = ""
                            cur_batch_id += 1
                            cur_batch_size = 0
                            batch_total_time = 0
                            dep_job_id_list = []
                            cases_list = []
                            slurm_batch_file.close()

                        # complete batch info as previous one was submitted
                        if submit_prev:
                            # append content of batch command with run of the case
                            batch_cmd += case.build_run_batch(mem_log=self.__mem_log)
                            cur_batch_size += 1
                            batch_total_time += float(case.expected_time)
                            cases_list.append(case)
                            if level > 0:
                                depend_id = self.graph.graph_dict[case].job_id
                                if depend_id not in dep_job_id_list:
                                    dep_job_id_list.append(depend_id)

                if cur_batch_size > 0:
                    slurm_batch_name = "slurm_batch_file_" + \
                                       str(cur_batch_id) + ".sh"
                    slurm_batch_file = open(slurm_batch_name, mode='w')

                    # fill file with template
                    hh, mm = divmod(batch_total_time, 60)
                    cmd = slurm_batch_template.format(nproc+1,
                                                      math.ceil(hh),
                                                      math.ceil(mm),
                                                      cur_batch_id,
                                                      job_name)

                    # add exclusive option to batch template for
                    # computation with at least 6 processes
                    if nproc+1 > 5:
                        cmd += "#SBATCH --exclusive\n"

                    # add user defined options if needed
                    if self.__slurm_batch_args:
                        for _p in self.__slurm_batch_args:
                            cmd += "#SBATCH " + _p + "\n"

                    cmd += "\n"
                    slurm_batch_file.write(cmd)

                    # fill file with batch command for several cases
                    slurm_batch_file.write(batch_cmd)
                    slurm_batch_file.flush()

                    # submit batch
                    if level < 1:
                        output = subprocess.check_output(['sbatch',
                                                          slurm_batch_name])
                    else:
                        # list of dependency id should be in the
                        # :id1:id2:id3 format
                        list_id = ""
                        for item in dep_job_id_list:
                            list_id += ":" + str(item)
                        if len(list_id) > 0:
                            output = subprocess.check_output(['sbatch',
                                     "--dependency=afterany"
                                     + list_id, slurm_batch_name])
                        else:
                            # empty list can occur with existing runs in study
                            output = subprocess.check_output(['sbatch',
                                     slurm_batch_name])

                    # find job id with regex and store it
                    msg = output.decode('utf-8').strip()
                    match = re.search('\d{8}', msg)
                    job_id = match.group()
                    tot_job_id_list.append(job_id)
                    self.reporting('    - %s ...' % msg)
                    for elem in cases_list:
                        elem.job_id = job_id

                    batch_cmd = ""
                    cur_batch_id += 1
                    cur_batch_size = 0
                    batch_total_time = 0
                    dep_job_id_list = []
                    cases_list = []
                    slurm_batch_file.close()

        # final submission for postprocessing, comparaison and state analysis
        slurm_batch_name = "slurm_batch_file_" + str(cur_batch_id) + ".sh"
        slurm_batch_file = open(slurm_batch_name, mode='w')

        # fill file with template
        # we consider 10 minutes per study
        hh, mm = divmod(self.n_study*10, 60)
        cmd = slurm_batch_template.format(1,
                                          math.ceil(hh),
                                          math.ceil(mm),
                                          cur_batch_id,
                                          job_name)

        # add user defined options if needed
        if self.__slurm_batch_args:
            for _p in self.__slurm_batch_args:
                cmd += "#SBATCH " + _p + "\n"

        cmd += "\n"
        slurm_batch_file.write(cmd)

        # fill file with batch command for state analysis
        batch_cmd += self.build_final_batch(self.__postpro,
                                            self.__compare,
                                            self.__sheet,
                                            state_file_name)
        slurm_batch_file.write(batch_cmd)
        slurm_batch_file.flush()

        # list of dependency id should be in the
        # :id1:id2:id3 format
        list_id = ""
        for item in tot_job_id_list:
            list_id += ":" + str(item)
        if len(list_id) > 0:
            output = subprocess.check_output(['sbatch',"--dependency=afterany"
                   + list_id, slurm_batch_name])
        else:
            # empty list can occur with existing runs in study
            output = subprocess.check_output(['sbatch', slurm_batch_name])

        msg = output.decode('utf-8').strip()
        self.reporting('    - %s ...' % msg)

        slurm_batch_file.close()

        os.chdir(self.__dest)

        self.reporting('')

    #---------------------------------------------------------------------------

    def build_final_batch(self, postpro, compare, report, state_file_name):
        """
        Launch state option in DESTINATION
        """

        final_cmd = "cd " + self.__dest + os.linesep

        e = os.path.join(self.__pkg.get_dir('bindir'), self.__exe)

        # To handle possible back-end issues for reports generation
        # such as unavailable packages, we allow to define a specific
        # back-end executable for postprocessing, which may be different from
        # the main executable (such as one in a container with required
        # prerequisites)
        config = configparser.ConfigParser()
        config.read(self.__pkg.get_configfiles())
        if config.has_option('studymanager', 'postprocessing_exec'):
            e = config.get('studymanager', 'postprocessing_exec')

        # final analysis after all run_cases are finished
        final_cmd += e + " smgr --state" \
                       + " -f " + self.__filename \
                       + " --repo " + self.__repo \
                       + " --dest " + self.__dest \
                       + " --state-file " + state_file_name
        if postpro:
            final_cmd += " --post"
        if compare:
            final_cmd += " --compare"
        if report:
            final_cmd += " --report"

        # add tags options
        if self.__with_tags:
            tags = ','.join(str(n) for n in self.__with_tags)
            final_cmd += " --with-tags " + '"' + tags + '"'
        if self.__without_tags:
            tags = ','.join(str(n) for n in self.__without_tags)
            final_cmd += " --without-tags " + '"' + tags + '"'

        return final_cmd

    #---------------------------------------------------------------------------

    def report_state(self, state_file_name):
        """
        Report state of all cases.
        Warning, if the markup of the case is repeated in the xml file of parameters,
        the run of the case is also repeated.
        """

        self.reporting("  o Check state for all cases")

        run_timeout = 3600

        s_prev = ""
        c_prev = ""

        add_header_and_footer = True

        colors = {case_state.UNKNOWN: "rgb(227,218,201)",
                  case_state.STAGING: "rgb(255,191,0)",
                  case_state.STAGED: "rgb(176,196,222)",
                  case_state.PREPROCESSED: "rgb(216,191,216)",
                  case_state.COMPUTED: "rgb(120,213,124)",
                  case_state.FINALIZING: "rgb(127,255,0)",
                  case_state.FINALIZED: "rgb(120,213,124)",
                  case_state.RUNNING: "rgb(65,105,225)",
                  case_state.EXCEEDED_TIME_LIMIT: "rgb(0,255,255)",
                  case_state.FAILED: "rgb(250,128,114)"}

        messages = {case_state.UNKNOWN: "Unknown",
                    case_state.STAGING: "Staging",
                    case_state.STAGED: "Staged",
                    case_state.PREPROCESSED: "Preprocesses",
                    case_state.COMPUTED: "OK",
                    case_state.FINALIZING: "Finalizing",
                    case_state.FINALIZED: "OK",
                    case_state.RUNNING: "Running",
                    case_state.EXCEEDED_TIME_LIMIT: "Time limit",
                    case_state.FAILED: "FAILED"}

        fd = open(state_file_name, 'w')

        if add_header_and_footer:
            fd.write("<html>\n")
            fd.write("<head></head>\n")
            fd.write("<body>\n")

        fd.write("<br><i><u>Destination folder:</u></i> "+ self.__dest + "</br> </br>")
        fd.write("Case states:\n\n")
        t = "<table width=100% cellspacing=\"2\">\n"
        t += "<tr class=\"top\">"

        t += "<td width=18%>Study</td>"
        t += "<td width=12%>Case</td>"
        t += "<td width=11%>Run id</td>"
        t += "<td width=12%>State</td>"
        t += "<td width=8% align=\"right\">Time (s)</td>"
        t += "<td width=9% align=\"right\">CPU (s)</td>"
        t += "<td width=8% align=\"right\">Prepro (s)</td>"
        t += "<td width=10% align=\"right\">Mem max (Mb)</td>"
        t += "<td width=6% align=\"right\">N ranks</td>"
        t += "<td width=6% align=\"right\">N threads</td>"
        t += "</tr>\n"

        fd.write(t)

        for case in self.graph.graph_dict:

            state, info = case.get_state(run_timeout=run_timeout)

            for k in info.keys():
                if info[k] is None:
                    info[k] = ''

            self.reporting(case.title, str(state), str(info))

            s = case.study
            c = case.label

            m = info['message']
            if not m:
                m = messages[state]

            mem_max = -1
            mem_s = ''
            try:
                for k in ('compute_mem', 'preproces_mem'):
                    sk = info[k]
                    if sk:
                        mem_max = max(float(sk)/1024., mem_max)
                        mem_s = str(round(mem_max))
            except Exception:
                pass

            n_leaks = None
            try:
                n_leaks = int(info['memory_leaks'])
                if n_leaks > 0:
                    colors[case_state.COMPUTED] = "rgb(0,106,62)"
                    colors[case_state.FINALIZING] = "rgb(4,128,0)"
                    colors[case_state.FINALIZED] = "rgb(0,106,62)"
                    m += " (memory leaks)"

            except Exception:
                pass

            prepro_time = ""
            try:
                prepro_time = '{:g}'.format(float(info['preprocess_time']))
            except Exception:
                pass

            t = ""
            if s != s_prev:
                t += "<tr class=\"top\"><td>" + s + "</td>"
                t += "<td>" + c + "</td>"
            else:
                t += "<tr><td> </td>\n"
                if c != c_prev:
                    t += "<td>" + c + "</td>"
                else:
                    t += "<td> </td>\n"

            t += "<td>" + case.run_id + "</td>"
            t += "<td style=\"background-color:" + colors[state] + "\">" + m + "</td>"
            t += "<td align=\"right\">" + str(info['compute_time']) + "</td>"
            t += "<td align=\"right\">" + str(info['compute_time_usage']) + "</td>"
            t += "<td align=\"right\">" + prepro_time + "</td>"
            t += "<td align=\"right\">" + mem_s + "</td>"
            t += "<td align=\"right\">" + str(info['mpi_ranks']) + "</td>"
            t += "<td align=\"right\">" + str(info['omp_threads']) + "</td>"
            t += "</tr>\n"
            fd.write(t)

            s_prev = s
            c_prev = c

        t = "</table>\n</br>\n"
        fd.write(t)

        if add_header_and_footer:
            fd.write("</body>\n")
            fd.write("</html>\n")

        fd.close()

        self.reporting('')

    #---------------------------------------------------------------------------

    def check_compare(self):
        """
        Check coherency between xml file of parameters and repository.
        Stop if you try to make a comparison with a file which does not exist.
        """
        check_msg = "  o Check compare of cases"
        self.reporting(check_msg)
        for case in self.graph.graph_dict:

            # reference directory passed in studymanager command line overwrites
            # destination in all cases (even if compare is defined by a compare
            # markup with a non empty destination)

            ref = None
            if self.__ref:
                ref = os.path.join(self.__ref, case.study)

            status, nodes, repo, dest, threshold, args = self.__parser.getCompare(case.node)
            for i in range(len(nodes)):
                if status[i] and case.compare:
                    msg = case.check_dirs(nodes[i], repo[i], dest[i], reference=ref)
                    if msg:
                        self.reporting(msg)
                        case.compare = False
                        msg2 = "    - Case %s --> COMPARISON DISABLED" %(case.title)
                        self.reporting(msg2)

    #---------------------------------------------------------------------------

    def compare_case_and_report(self, case, repo, dest, threshold, args,
                                reference=None):
        """
        Compare the results for one computation and report
        """

        if case.compare:
            diff_value, m_size_eq = case.runCompare(self,
                                                    repo, dest,
                                                    threshold, args,
                                                    reference=reference)
            case.diff_value += diff_value
            case.m_size_eq = case.m_size_eq and m_size_eq

            if args:
                s_args = 'with args: %s' % args
            else:
                s_args = 'default mode'

            if not m_size_eq:
                self.reporting('    - compare %s (%s) --> DIFFERENT MESH SIZES FOUND'
                               %(case.title, s_args))
            elif diff_value:
                self.reporting('    - compare %s (%s) --> DIFFERENCES FOUND'
                               %(case.title, s_args))
            else:
                self.reporting('    - compare %s (%s) --> NO DIFFERENCES FOUND'
                               %(case.title, s_args))


    #---------------------------------------------------------------------------

    def compare(self):
        """
        Compare the results of the new computations with those from the
        Repository.
        """
        self.reporting('  o Compare cases')
        for case in self.graph.graph_dict:
            # reference directory passed in studymanager command line
            # overwrites destination in all cases (even if compare is
            # defined by a compare markup with a non empty destination)
            ref = None
            if self.__ref:
                ref = os.path.join(self.__ref, case.study)
            status, nodes, repo, dest, t, args = \
                                 self.__parser.getCompare(case.node)

            if status:
                for i in range(len(nodes)):
                    if status[i] and case.compare:
                        self.compare_case_and_report(case,
                                                     repo[i],
                                                     dest[i],
                                                     t[i],
                                                     args[i],
                                                     reference=ref)
            elif case.compare:
                 repo = ""
                 dest = ""
                 t    = None
                 args = None
                 self.compare_case_and_report(case,
                                              repo,
                                              dest,
                                              t,
                                              args,
                                              reference=ref)

        self.reporting('')

    #---------------------------------------------------------------------------

    def check_script(self):
        """
        Check coherency between xml file of parameters and repository.
        Stop if you try to run a script with a file which does not exist.
        """
        check_msg = "  o Check scripts of cases"
        self.reporting(check_msg)
        scripts_checked = False
        for case in self.graph.graph_dict:
            # search for scripts to check
            status, label, nodes, args, repo, dest = \
                    self.__parser.getScript(case.node)

            # check existing scripts
            for i in range(len(nodes)):
                if status[i] and case.plot:
                    post_folder = os.path.join(self.__dest, case.study, "POST")
                    # no dest for script in POST
                    msg = case.check_file(post_folder, "", label[i])
                    scripts_checked = True
                    if msg:
                        self.reporting(msg)
                        case.plot = False
                        msg2 = "    - Case %s --> POST DISABLED" %(case.title)
                        self.reporting(msg2)

        if scripts_checked:
            self.reporting('')

        return scripts_checked

    #---------------------------------------------------------------------------

    def scripts(self):
        """
        Launch external additional scripts with arguments for a case.
        Related to keyword script in smgr xml file
        """
        # create smgr_post_pro.log in dest/STUDY
        self.reporting("  o Run scripts of cases:")
        for case in self.graph.graph_dict:
            status, label, nodes, args, repo, dest = self.__parser.getScript(case.node)
            for i in range(len(label)):
                if status[i] and case.plot:
                    cmd = os.path.join(self.__dest, case.study, "POST", label[i])
                    script_name = label[i]
                    if os.path.isfile(cmd):
                        # ensure script is executable
                        set_executable(cmd)

                        if args[i]:
                            cmd += " " + args[i]

                        # folder should not finished by / to prevent error in scripts
                        folder = case.run_dir
                        if dest[i]:
                            folder = os.path.join(case.run_dir, dest[i])
                        cmd += " -d " + folder

                        retcode, t = run_studymanager_command(cmd,
                                                              self.__log_post_file)
                        stat = "FAILED" if retcode != 0 else "OK"

                        self.reporting('    - script %s %s in %s (%s s)'
                                       %(script_name, stat, case.title, t))

                    else:
                        self.reporting('    - script %s not found' % cmd)

        self.reporting('')

    #---------------------------------------------------------------------------

    def postpro(self):
        """
        Launch external additional scripts with arguments for studies.
        Related to keyword postpro in smgr xml file
        """

        previous_study_name = None
        for case in self.graph.graph_dict:

            # detect new study
            if case.study is not previous_study_name:
                previous_study_name = case.study

                script, label, nodes, args = self.__parser.getPostPro(case.study)

                for i in range(len(label)):
                    if i == 0:
                        self.reporting('  o Postprocessing results of study: ' + case.study)
                    if script[i]:
                        cmd = os.path.join(self.__dest, case.study, "POST", label[i])
                        if os.path.isfile(cmd):
                            sc_name = os.path.basename(cmd)
                            # ensure script is executable
                            set_executable(cmd)

                            # retrieve study_object from index
                            study_label, study_object = self.studies[case.study_index]

                            list_cases, list_dir = study_object.getRunDirectories()
                            cmd += ' ' + args[i] + ' -c "' + list_cases + '" -d "' \
                                   + list_dir + '" -s ' + case.study

                            self.reporting('    - running postpro %s' % sc_name,
                                           stdout=True, report=False, status=True)

                            retcode, t = run_studymanager_command(cmd, self.__log_post_file)
                            stat = "FAILED" if retcode != 0 else "OK"

                            self.reporting('    - postpro %s --> %s (%s s)' \
                                           % (stat, sc_name, t))

                        else:
                            self.reporting('    - postpro %s not found' % cmd)

        # erase empty log file
        self.__log_post_file.close()
        log_post_file = open(self.__log_post_name, mode='r')
        content = log_post_file.read()
        log_post_file.close()
        if not content:
            os.system("rm -f " + self.__log_post_name)
        else:
            self.reporting(" Warning: logs generated during post. See %s\n" \
                           %self.__log_post_name, stdout=True, report=True)

        self.reporting('')

    #---------------------------------------------------------------------------

    def check_data(self, case):
        """
        Check coherency between xml file of parameters and destination
        for data markups of a run.
        """
        for node in self.__parser.getChildren(case.node, "data"):

            plots, file_name, dest, repo = self.__parser.getResult(node)
            # we check in monitoring and profiles folders if no dest is given
            if dest:
                msg = case.check_file(case.run_dir, dest, file_name)
            else:
                for sd in ("monitoring", "profiles", "."):
                    msg = case.check_file(case.run_dir, sd, file_name)
                    # stop if file is found
                    if not msg:
                        break

            if msg:
                self.reporting(msg)
                return False

        return True

    #---------------------------------------------------------------------------

    def check_probes(self, case):
        """
        Check coherency between xml file of parameters and destination
        for probes markups of a run.
        """
        for node in self.__parser.getChildren(case.node, "probes"):
            file_name, dest, fig = self.__parser.getProbes(node)
            # probes are in monitoring folder
            dest = "monitoring/" + dest
            msg = case.check_file(case.run_dir, dest, file_name)
            if msg:
                self.reporting(msg)
                return False

        return True

    #---------------------------------------------------------------------------

    def check_input(self, case):
        """
        Check coherency between xml file of parameters and destination
        for input markups of a run.
        """
        for node in self.__parser.getChildren(case.node, "input"):
            file_name, dest, repo, tex = self.__parser.getInput(node)
            msg = case.check_file(case.run_dir, dest, file_name)
            if msg:
                self.reporting(msg)

    #---------------------------------------------------------------------------

    def check_plots_and_input(self):
        """
        Check coherency between xml file of parameters and repository.
        Stop if you try to make a plot of a file which does not exist.
        """
        check_msg = "  o Check plots and input of cases"
        self.reporting(check_msg)
        for case in self.graph.graph_dict:
            if case.plot:
                # verify input, data and probes
                self.check_input(case)
                found_data = self.check_data(case)
                found_probes = self.check_probes(case)
                # stop if data or probes are missing
                if not found_data or not found_probes:
                    case.plot = False
                    msg = "    - Case %s --> POST DISABLED" %(case.title)
                    self.reporting(msg)

        self.reporting('')

    #---------------------------------------------------------------------------

    def plot(self):
        """
        Plot data.
        """

        previous_study_name = None
        previous_study_object = None
        list_cases = []
        for case in self.graph.graph_dict:

            # detect new study
            if case.study is not previous_study_name:

                # submit previous study
                if previous_study_name and list_cases:
                    self.reporting('  o Plot study: ' + previous_study_name)
                    error = self.__plotter.plot_study(self.__dest,
                                                      previous_study_object,
                                                      list_cases,
                                                      self.__dis_tex,
                                                      self.__default_fmt)
                    if error:
                        self.reporting(error)
                    list_cases = []

                # store actual study name and object
                study_label, study_object = self.studies[case.study_index]
                previous_study_name = study_label
                previous_study_object = study_object

            # add case in list
            list_cases.append(case)

        # final study
        if previous_study_name and list_cases:
            self.reporting('  o Plot study: ' + previous_study_name)
            error = self.__plotter.plot_study(self.__dest,
                                              previous_study_object,
                                              list_cases,
                                              self.__dis_tex,
                                              self.__default_fmt)
            if error:
                self.reporting(error)

        self.reporting('')

    #---------------------------------------------------------------------------

    def report_input(self, doc, i_nodes, s_label, c_label=None, r_label="RESU",
                     run_label=None):
        """
        Add input to report detailed.
        """
        for i_node in i_nodes:
            f, dest, repo, tex = self.__parser.getInput(i_node)
            doc.appendLine("\\subsubsection{%s}" % f)

            if dest:
                d = dest
                dd = self.__dest
            elif repo:
                d = repo
                dd = self.__repo
            else:
                d = ""
                dd = ""

            if c_label:
                fd = os.path.join(dd, s_label, c_label, r_label, run_label)
            else:
                fd = os.path.join(dd, s_label, 'POST')

            if d == '':
                ff = os.path.join(fd, f)
                if not os.path.isfile(ff):
                    if os.path.isdir(fd):
                        l = os.listdir(fd)
                        if len(l) == 1:
                            ff = os.path.join(fd, l[0], f)
            else:
                ff = os.path.join(fd, d, f)

            if not os.path.isfile(ff):
                self.reporting("    Warning: this file does not exist: %s" %ff)
            elif ff[-4:] in ('.png', '.jpg', '.pdf') or ff[-5:] == '.jpeg':
                doc.addFigure(ff)
            elif tex == 'on':
                doc.addTexInput(ff)
            else:
                doc.addInput(ff)

    #---------------------------------------------------------------------------

    def copy_input(self, i_nodes, s_label, c_label, run_label, r_label="RESU"):
        """
        Copy input in POST for later description report generation
        """

        # list of accepted formats
        input_format = ['.png', '.jpg', '.pdf', '.tex', '.jpeg']

        for i_node in i_nodes:
            file_name, tmp, repo, tex = self.__parser.getInput(i_node)
            input_file = os.path.join(self.__dest, s_label, c_label, r_label,
                                      run_label, file_name)

            # Input copied in POST/"CURRENT"/CASE/run_id folder
            # file_name can include another folder (ex: datasets/fig.png)
            file_dest = os.path.join(self.__dest, s_label, 'POST', 'CURRENT',
                                     c_label, run_label, file_name)
            dest_folder = os.path.dirname(file_dest)

            if os.path.isfile(input_file):
                file_format = input_file[-4:]
                if input_file[-5] == ".":
                    file_format = input_file[-5:]
                if file_format in input_format:
                    if not os.path.exists(dest_folder):
                        os.makedirs(dest_folder)
                    shutil.copyfile(input_file, file_dest)

    #---------------------------------------------------------------------------

    def build_reports(self, report_fig):
        """
        @type report_fig: C{String}
        @param report_fig: name of the figures report.
        @rtype: C{List} of C{String}
        @return: list of file to be attached to the report.
        """
        attached_files = []

        self.reporting('  o Generation of the automatic detailed report')

        # figures report
        doc = Report(self.__dest,
                     report_fig,
                     self.__pdflatex)

        previous_study_name = None
        for case in self.graph.graph_dict:

            # detect new study
            if case.study is not previous_study_name:
                # retrieve study_object from index
                study_label, study_object = self.studies[case.study_index]
                previous_study_name = study_label
                doc.appendLine("\\section{%s}" % case.study)

                # handle figures and inputs
                if study_object.matplotlib_figures or study_object.input_figures:
                    doc.appendLine("\\subsection{Graphical results}")
                    for g in study_object.matplotlib_figures:
                        doc.addFigure(g)
                    for g in study_object.input_figures:
                        doc.addFigure(g)

                # handle the input nodes that are inside postpro nodes
                script, label, nodes, args = self.__parser.getPostPro(case.study)

                needs_pp_input = False
                for i in range(len(label)):
                    if script[i]:
                        input_nodes = \
                            self.__parser.getChildren(nodes[i], "input")
                        if input_nodes:
                            needs_pp_input = True
                            break

                if needs_pp_input:
                    doc.appendLine("\\subsection{Results for "
                               "post-processing cases}")
                    for i in range(len(label)):
                        if script[i]:
                            input_nodes = \
                                self.__parser.getChildren(nodes[i], "input")
                            if input_nodes:
                                self.report_input(doc, input_nodes, case.study)

            # handle input nodes that are inside case nodes
            if case.plot:
                nodes = self.__parser.getChildren(case.node, "input")
                if nodes:
                    doc.appendLine("\\subsection{Results for "
                                   "case %s}" % case.label)
                    self.report_input(doc, nodes, case.study, case.label,
                                      case.resu, case.run_id)
                    # copy input in POST (TODO: should be moved)
                    self.copy_input(nodes, case.study, case.label, case.run_id)

        attached_files.append(doc.close())

        self.reporting('')

        return attached_files

    #---------------------------------------------------------------------------

    def build_description_report(self):
        """
        Generate description report of the study
        REPORT ans STYLE folders are mandatory
        """

        # save current location
        save_dir = os.getcwd()

        self.reporting('  o Generation of V&V description report')

        previous_study_name = None
        for case in self.graph.graph_dict:

            # detect new study
            if case.study is not previous_study_name:
                previous_study_name = case.study

                # change directory to make report pdf file
                make_dir = os.path.join(self.__dest, case.study, "REPORT")
                if os.path.isdir(make_dir):
                    os.chdir(make_dir)

                    # Generation of keywords and/or readme tex files
                    # based on metadata within the .xml file
                    from code_saturne.studymanager.cs_studymanager_metadata import study_metadata
                    md = study_metadata(self.__pkg, os.path.join(self.__dest, self.__filename))
                    md.dump_keywords()
                    md.dump_readme()

                    log_name = "make_report_" + case.study + ".log"
                    log_path = os.path.join(self.__dest, log_name)
                    log_pdf = open(log_path, mode='w')
                    cmd = "make pdf"
                    pdf_retval, t = run_studymanager_command(cmd, log_pdf)
                    log_pdf.close()

                    report_pdf = "write-up.pdf"
                    if os.path.isfile(report_pdf):
                        self.reporting('    - write-up.pdf file was generated ' + \
                                       'in ' + case.study + "/REPORT folder.")
                        os.remove(log_path)
                    else:
                        self.reporting('    /!\ ERROR: write-up.pdf file was not ' + \
                                       'generated. See ' + log_path)

                else:
                    self.reporting('    - No REPORT folder: generation of ' + \
                                   'description file is aborted.')

        # move to initial location
        os.chdir(save_dir)

#-------------------------------------------------------------------------------
# class dependency_graph
#-------------------------------------------------------------------------------

class dependency_graph(object):

    def __init__(self):
        """ Initializes a dependency graph object to an empty dictionary
        """
        self.graph_dict = OrderedDict()
        # maximum number of level in the graph
        self.max_level = 0
        # maximum number of proc used in a case of the graph
        self.max_proc = 0

    def add_dependency(self, dependency):
        """ Defines dependency between two cases as an edge in the graph
        """
        (node1, node2) = dependency
        # only one dependency is possible
        if not self.graph_dict[node1]:
            self.graph_dict[node1] = node2
        else:
            msg = "Error in dependency graph" + node1.title + " can only" \
                + " have one dependency.\n"
            return msg

    def add_node(self, case):
        """ Add a case in the graph if not already there.
            Add a dependency when depends parameters is defined
        """
        msg = None
        if case not in self.graph_dict:
            self.graph_dict[case] = None

            if case.depends:
                for neighbor in self.graph_dict:
                    neighbor_name = neighbor.study + '/' + neighbor.label \
                                    + '/' + neighbor.resu + '/' + neighbor.run_id
                    if neighbor_name == case.depends:
                        # cases with dependency are level > 0 and connected to the dependency
                        self.add_dependency((case, neighbor))
                        case.level = neighbor.level + 1
                        self.max_level = max([self.max_level, case.level])
                        break
                if case.level is None:
                    case.level = 0
                    msg = "Dependency " + case.depends + " was not found in the list " \
                          + "of considered cases. " + case.title + " will be considered " \
                          + "without dependency in the graph. Please check that the " \
                          + "required results exist.\n"

            else:
                # cases with no dependency are level 0
                case.level = 0

            self.max_proc = max([self.max_proc, int(case.n_procs)])
        return msg

    def nodes(self):
        """ returns the cases of the dependency graph """
        return list(self.graph_dict.keys())

    def dependencies(self):
        """ returns the dependencies between the cases of the graph """
        dependencies = []
        for node in self.graph_dict:
            for neighbor in self.graph_dict[node]:
                if (neighbor, node) not in dependencies:
                    dependencies.append((node, neighbor))
        return dependencies

    def extract_sub_graph(self, filter_level, filter_n_procs):
        """ extracts a sub_graph based on level and n_procs criteria"""
        sub_graph = dependency_graph()
        for node in self.graph_dict:
            keep_node_l = True
            keep_node_p = True
            if filter_level is not None:
                keep_node_l = int(node.level) == int(filter_level)
            if filter_n_procs is not None:
                keep_node_p = int(node.n_procs) == int(filter_n_procs)
            if keep_node_l and keep_node_p:
                msg = sub_graph.add_node(node)
                if msg:
                    self.reporting("Warning in sub graph: " + msg)
        return sub_graph

    def __str__(self):
        res = "\nList of cases: "
        for node in self.nodes():
            res += str(node.title) + " "
        res += "\nList of dependencies: "
        for dependency in self.dependencies():
            (node1, node2) = dependency
            res += '\n' + str(node1.title) + ' depends on ' + str(node2.title)
        return res

#-------------------------------------------------------------------------------
