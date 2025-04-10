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

"""
This module defines the 'Main fields' page.

This module contains the following classes:
- LabelDelegate
- NatureDelegate
- EnthalpyDelegate
- CriterionDelegate
- CarrierDelegate
- StandardItemModelMainFields
- MainFieldsView
"""

#-------------------------------------------------------------------------------
# Library modules import
#-------------------------------------------------------------------------------

import os, sys, string, types
import logging

#-------------------------------------------------------------------------------
# Third-party modules
#-------------------------------------------------------------------------------

from code_saturne.gui.base.QtCore    import *
from code_saturne.gui.base.QtGui     import *
from code_saturne.gui.base.QtWidgets import *

#-------------------------------------------------------------------------------
# Application modules import
#-------------------------------------------------------------------------------

from code_saturne.gui.base.QtPage import ComboModel, RegExpValidator
from code_saturne.gui.base.QtPage import from_qvariant, to_text_string
from code_saturne.model.Common import LABEL_LENGTH_MAX, GuiParam

from code_saturne.gui.case.MainFields import Ui_MainFields
from code_saturne.model.MainFieldsModel import MainFieldsModel

from code_saturne.model.LagrangianModel import LagrangianModel

#-------------------------------------------------------------------------------
# log config
#-------------------------------------------------------------------------------

logging.basicConfig()
log = logging.getLogger("MainFieldsView")
log.setLevel(GuiParam.DEBUG)

#-------------------------------------------------------------------------------
# Line edit delegate for the label
#-------------------------------------------------------------------------------

class LabelDelegate(QItemDelegate):
    """
    Use of a QLineEdit in the table.
    """
    def __init__(self, parent=None):
        QItemDelegate.__init__(self, parent)
        self.parent = parent
        self.old_plabel = ""


    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        self.old_label = ""
        rx = "[_a-zA-Z][_A-Za-z0-9]{1," + str(LABEL_LENGTH_MAX-1) + "}"
        self.regExp = QRegExp(rx)
        v = RegExpValidator(editor, self.regExp)
        editor.setValidator(v)
        return editor


    def setEditorData(self, editor, index):
        editor.setAutoFillBackground(True)
        value = from_qvariant(index.model().data(index, Qt.DisplayRole), to_text_string)
        self.old_plabel = str(value)
        editor.setText(value)


    def setModelData(self, editor, model, index):
        if not editor.isModified():
            return

        if editor.validator().state == QValidator.Acceptable:
            new_plabel = str(editor.text())

            if new_plabel in model.mdl.getFieldLabelsList():
                default = {}
                default['label']  = self.old_plabel
                default['list']   = model.mdl.getFieldLabelsList()
                default['regexp'] = self.regExp
                log.debug("setModelData -> default = %s" % default)

                from code_saturne.gui.case.VerifyExistenceLabelDialogView import VerifyExistenceLabelDialogView
                dialog = VerifyExistenceLabelDialogView(self.parent, default)
                if dialog.exec_():
                    result = dialog.get_result()
                    new_plabel = result['label']
                    log.debug("setModelData -> result = %s" % result)
                else:
                    new_plabel = self.old_plabel

            model.setData(index, new_plabel, Qt.DisplayRole)

#-------------------------------------------------------------------------------
# Combo box delegate for the nature of field
#-------------------------------------------------------------------------------

class NatureDelegate(QItemDelegate):
    """
    Use of a combo box in the table.
    """
    def __init__(self, parent):
        super(NatureDelegate, self).__init__(parent)
        self.parent   = parent


    def createEditor(self, parent, option, index):
        editor = QComboBox(parent)
        self.modelCombo = ComboModel(editor, 3, 1)
        self.modelCombo.addItem(self.tr("liquid"), 'liquid')
        self.modelCombo.addItem(self.tr("gas"), 'gas')
        self.modelCombo.addItem(self.tr("solid"), 'solid')

        row = index.row()
        if (row == 0) :
            self.modelCombo.disableItem(2)
        else :
            self.modelCombo.enableItem(2)

        editor.installEventFilter(self)
        return editor


    def setEditorData(self, comboBox, index):
        col = index.column()
        string = index.model().getData(index)[col]
        self.modelCombo.setItem(str_model=string)


    def setModelData(self, comboBox, model, index):
        txt = str(comboBox.currentText())
        value = self.modelCombo.dicoV2M[txt]
        log.debug("NatureDelegate value = %s"%value)

        selectionModel = self.parent.selectionModel()
        for idx in selectionModel.selectedIndexes():
            if idx.column() == index.column():
                model.setData(idx, value, Qt.DisplayRole)


#-------------------------------------------------------------------------------
# Combo box delegate for the enthalpy
#-------------------------------------------------------------------------------

class EnthalpyDelegate(QItemDelegate):
    """
    Use of a combo box in the table.
    """
    def __init__(self, parent, mdl):
        super(EnthalpyDelegate, self).__init__(parent)
        self.parent   = parent
        self.mdl      = mdl


    def createEditor(self, parent, option, index):
        editor = QComboBox(parent)
        predefined_flow = self.mdl.getPredefinedFlow()
        if predefined_flow in ["free_surface", "boiling_flow", "droplet_flow", "multiregime"]:
            self.modelCombo = ComboModel(editor, 2, 1)
            self.modelCombo.addItem(self.tr("off"), 'off')
            self.modelCombo.addItem(self.tr("total enthalpy"), 'total_enthalpy')
        else:
            self.modelCombo = ComboModel(editor, 3, 1)
            self.modelCombo.addItem(self.tr("off"), 'off')
            self.modelCombo.addItem(self.tr("total enthalpy"), 'total_enthalpy')
            self.modelCombo.addItem(self.tr("specific enthalpy"), 'specific_enthalpy')

        editor.installEventFilter(self)
        return editor


    def setEditorData(self, comboBox, index):
        col = index.column()
        string = index.model().getData(index)[col]
        self.modelCombo.setItem(str_model=string)


    def setModelData(self, comboBox, model, index):
        txt = str(comboBox.currentText())
        value = self.modelCombo.dicoV2M[txt]
        log.debug("EnthalpyDelegate value = %s"%value)

        selectionModel = self.parent.selectionModel()
        for idx in selectionModel.selectedIndexes():
            if idx.column() == index.column():
                model.setData(idx, value, Qt.DisplayRole)


#-------------------------------------------------------------------------------
# Combo box delegate for the criterion of field
#-------------------------------------------------------------------------------

class CriterionDelegate(QItemDelegate):
    """
    Use of a combo box in the table.
    """
    def __init__(self, parent):
        super(CriterionDelegate, self).__init__(parent)
        self.parent   = parent


    def createEditor(self, parent, option, index):
        editor = QComboBox(parent)
        self.modelCombo = ComboModel(editor, 3, 1)
        self.modelCombo.addItem(self.tr("continuous"), 'continuous')
        self.modelCombo.addItem(self.tr("dispersed"), 'dispersed')
        self.modelCombo.addItem(self.tr("auto"), 'auto')
        # TODO to delete if/when the auto option is implemented
        self.modelCombo.disableItem(2)
        # fixed to continuous for field 1
        if index.row() == 0 :
            editor.setEnabled(False)

        editor.installEventFilter(self)
        return editor


    def setEditorData(self, comboBox, index):
        row = index.row()
        col = index.column()
        string = index.model().getData(index)[col]
        self.modelCombo.setItem(str_model=string)


    def setModelData(self, comboBox, model, index):
        txt = str(comboBox.currentText())
        value = self.modelCombo.dicoV2M[txt]
        log.debug("CriterionDelegate value = %s"%value)

        selectionModel = self.parent.selectionModel()
        for idx in selectionModel.selectedIndexes():
            if idx.column() == index.column():
                model.setData(idx, value, Qt.DisplayRole)


#-------------------------------------------------------------------------------
# Combo box delegate for the carrier field
#-------------------------------------------------------------------------------


class CarrierDelegate(QItemDelegate):
    """
    Use of a combo box in the table.
    """
    def __init__(self, parent, mdl):
        super(CarrierDelegate, self).__init__(parent)
        self.parent   = parent
        self.mdl      = mdl


    def createEditor(self, parent, option, index):
        editor = QComboBox(parent)
        self.modelCombo = ComboModel(editor, 1, 1)
        field = self.mdl.list_of_fields[index.row()]
        if field.flow_type == "continuous" :
            self.modelCombo.addItem(self.tr("off"), 'off')
        else :
            for fld in self.mdl.getContinuousFieldList() :
                self.modelCombo.addItem(self.tr(fld.label), fld.label)
            if field.phase == "solid":
                self.modelCombo.addItem(self.tr("all"), "all")

        editor.installEventFilter(self)
        return editor


    def setEditorData(self, comboBox, index):
        row = index.row()
        col = index.column()
        string = index.model().getData(index)[col]
        self.modelCombo.setItem(str_model=string)


    def setModelData(self, comboBox, model, index):
        txt = str(comboBox.currentText())
        value = self.modelCombo.dicoV2M[txt]
        log.debug("CarrierDelegate value = %s"%value)

        selectionModel = self.parent.selectionModel()
        for idx in selectionModel.selectedIndexes():
            if idx.column() == index.column():
                model.setData(idx, value, Qt.DisplayRole)


#-------------------------------------------------------------------------------
# StandardItemModelMainFields class
#-------------------------------------------------------------------------------

class StandardItemModelMainFields(QStandardItemModel):

    def __init__(self, mdl):
        """
        """
        QStandardItemModel.__init__(self)

        self.headers = [ self.tr("Field\nlabel"),
                         self.tr("Phase of\nfield"),
                         self.tr("Interfacial criterion"),
                         self.tr("Carrier phase"),
                         self.tr("Variable density"),
                         self.tr("Energy\nresolution")]

        self.setColumnCount(len(self.headers))

        self.tooltip = []

        self._data = []
        self.mdl = mdl


    def data(self, index, role):
        if not index.isValid():
            return None

        if role == Qt.ToolTipRole:
            return None

        elif role == Qt.DisplayRole:
            data = self._data[index.row()][index.column()]
            if index.column() in (0, 1, 2, 3, 5):
                if data:
                    return data
                else:
                    return None

        elif role == Qt.CheckStateRole:
            data = self._data[index.row()][index.column()]
            if index.column() == 4:
                if data == 'on':
                    return Qt.Checked
                else:
                    return Qt.Unchecked

        elif role == Qt.TextAlignmentRole:
            return Qt.AlignCenter

        return None


    def flags(self, index):

        if not index.isValid():
            return Qt.ItemIsEnabled
        if index.column() == 4:
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
        elif index.column() == 5:
            if self.mdl.getPredefinedFlow() != "None" \
                    and self.mdl.getPredefinedFlow() != "particles_flow" \
                    and self.mdl.getPhaseChangeTransferStatus() == "on":
                return Qt.ItemIsSelectable
            else:
                return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable
        elif index.column() == 1 or index.column() == 2:

            field = self.mdl.list_of_fields[index.row()]
            if self.mdl.getPredefinedFlow() != "None" and (index.row()==0 or index.row()==1):
                return Qt.ItemIsEnabled | Qt.ItemIsSelectable
            elif field.phase == "solid" and index.column() == 2:
                return Qt.ItemIsEnabled | Qt.ItemIsSelectable
            else:
                return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable

        else:
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable


    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.headers[section]
        return None


    def setData(self, index, value, role):
        if not index.isValid():
            return Qt.ItemIsEnabled

        # Update the row in the table
        row = index.row()
        col = index.column()
        field = self.mdl.list_of_fields[row]
        FieldId = field.f_id

        # Label
        if col == 0:
            new_plabel = from_qvariant(value, to_text_string)

            # Since all fields' labels are using the phase label, we rename
            # the labels in all formulas!
            old_plabel = self._data[row][col]
            for nf in self.mdl.case.xmlGetNodeList('formula'):
                if nf:
                    ftext = (str(nf).replace('<formula>','')).replace('</formula>','')
                    nf.xmlSetTextNode(ftext.replace(old_plabel, new_plabel))

            self._data[row][col] = new_plabel
            field.label = new_plabel
            self.updateItem()

        # Nature of field
        elif col == 1:
            new_nature = from_qvariant(value, to_text_string)
            self._data[row][col] = new_nature
            field.phase = new_nature
            self.updateItem()

        # Interfacial criterion
        elif col == 2:
            new_crit = from_qvariant(value, to_text_string)
            self._data[row][col] = new_crit
            self.mdl.setCriterion(FieldId, new_crit)
            # update carrier field
            self.updateItem()

        # Carrier field
        elif col == 3:
            new_carrier = from_qvariant(value, to_text_string)
            self._data[row][col] = new_carrier
            # set carrier field Id in XML
            if self._data[row][col] not in ["off", "all"] : # TODO move this test to model part ?
               fid = self.mdl.getFieldId(self._data[row][col])
            else :
               fid = self._data[row][col]
            field.carrier_id = fid

        # Compressible
        elif col == 4:
            state = from_qvariant(value, int)
            if state == Qt.Unchecked:
                self._data[row][col] = "off"
                field.compressible = "off"
            else:
                self._data[row][col] = "on"
                field.compressible = "on"

        # Energy resolution
        elif col == 5:
            state = from_qvariant(value, to_text_string)
            self._data[row][col] = state
            field.enthalpy_model = state

        self.dataChanged.emit(index, index)
        return True


    def getData(self, index):
        row = index.row()
        return self._data[row]


    def newItem(self, existing_fieldId=None):
        """
        Add/load a field in the model.
        """
        row = self.rowCount()

        field = self.mdl.addField(existing_fieldId)
        label = field.label
        nature = field.phase
        criterion = field.flow_type
        carrier = self.mdl.getFieldFromId(field.carrier_id)
        carrierLabel = carrier.label
        compressible = field.compressible
        energy       = field.enthalpy_model

        field = [label, nature, criterion, carrierLabel, compressible, energy]

        self._data.append(field)
        self.setRowCount(row+1)


    def loadItem(self, label, nature, criterion, carrierLabel, compressible, energy):
        """
        Add/load a field in the model.
        """
        row = self.rowCount()

        field = [label, nature, criterion, carrierLabel, compressible, energy]

        self._data.append(field)
        self.setRowCount(row+1)


    def updateItem(self):
        # update carrier field and criterion
        for i, field in enumerate(self.mdl.list_of_fields) :
            self._data[i][2] = field.flow_type
            # If field is continuous, ensure that carrier option is updated
            if field.flow_type == 'continuous':
                field.carrier_id = 'off'

            carrier_id = field.carrier_id
            self._data[i][3] = self.mdl.getFieldFromId(carrier_id).label


    def deleteItem(self, row):
        """
        Delete the row in the model.
        """
        del self._data[row]
        field_to_delete = self.mdl.list_of_fields[row]
        self.mdl.deleteField(field_to_delete.f_id)
        row = self.rowCount()
        self.setRowCount(row-1)
        self.updateItem()


#-------------------------------------------------------------------------------
# MainFieldsView class
#-------------------------------------------------------------------------------

class MainFieldsView(QWidget, Ui_MainFields):
    """
    Main fields layout.
    """
    def __init__(self, parent, case, tree):
        """
        Constructor
        """
        QWidget.__init__(self, parent)

        Ui_MainFields.__init__(self)
        self.setupUi(self)

        self.browser = tree
        self.case = case
        self.case.undoStopGlobal()
        self.mdl  = MainFieldsModel(self.case)
        self.lagr = LagrangianModel(self.case)

        # Main fields definition
        self.tableModelFields = StandardItemModelMainFields(self.mdl)
        self.tableViewFields.setModel(self.tableModelFields)
        self.tableViewFields.resizeColumnsToContents()
        self.tableViewFields.resizeRowsToContents()
        if QT_API == "PYQT4":
            self.tableViewFields.verticalHeader().setResizeMode(QHeaderView.ResizeToContents)
            self.tableViewFields.horizontalHeader().setResizeMode(QHeaderView.ResizeToContents)
            self.tableViewFields.horizontalHeader().setResizeMode(0,QHeaderView.Stretch)
        elif QT_API == "PYQT5":
            self.tableViewFields.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            self.tableViewFields.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            self.tableViewFields.horizontalHeader().setSectionResizeMode(0,QHeaderView.Stretch)
        self.tableViewFields.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tableViewFields.setSelectionMode(QAbstractItemView.SingleSelection)

        delegateLabel      = LabelDelegate(self.tableViewFields)
        delegateNature     = NatureDelegate(self.tableViewFields)
        delegateCriterion = CriterionDelegate(self.tableViewFields)
        delegateCarrier = CarrierDelegate(self.tableViewFields, self.mdl)
        delegateEnthalpy = EnthalpyDelegate(self.tableViewFields, self.mdl)

        self.tableViewFields.setItemDelegateForColumn(0, delegateLabel)
        self.tableViewFields.setItemDelegateForColumn(1, delegateNature)
        self.tableViewFields.setItemDelegateForColumn(2, delegateCriterion)
        self.tableViewFields.setItemDelegateForColumn(3, delegateCarrier)
        self.tableViewFields.setItemDelegateForColumn(5, delegateEnthalpy)

        model = self.mdl.getPredefinedFlow()
        if model != "None":
            self._hidePushButtons()
        else:
            self._initializePushButtons()

        for field in self.mdl.list_of_fields:
            label = field.label
            nature = field.phase
            criterion = field.flow_type
            carrier = field.carrier_id
            carrierLabel = self.mdl.getFieldFromId(carrier).label
            compressible = field.compressible
            energy = field.enthalpy_model
            self.tableModelFields.loadItem(label, nature, criterion, carrierLabel, compressible, energy)
        self.browser.configureTree(self.case)

        self.case.undoStartGlobal()

    def _initializePushButtons(self):
        self.pushButtonAdd.clicked.connect(self.slotAddField)
        self.pushButtonDelete.clicked.connect(self.slotDeleteField)
        self.tableModelFields.dataChanged.connect(self.dataChanged)

        if len(self.mdl.getFieldIdList()) > 2:
            self.pushButtonDelete.setEnabled(1)
        else:
            self.pushButtonDelete.setEnabled(0)

    def _hidePushButtons(self):
        self.pushButtonAdd.hide()
        self.pushButtonDelete.hide()

    def dataChanged(self, topLeft, bottomRight):
        for row in range(topLeft.row(), bottomRight.row() + 1):
            self.tableViewFields.resizeRowToContents(row)
        for col in range(topLeft.column(), bottomRight.column() + 1):
            self.tableViewFields.resizeColumnToContents(col)

        self.browser.configureTree(self.case)


    @pyqtSlot()
    def slotAddField(self):
        """
        Add a Field.
        """
        self.tableViewFields.clearSelection()
        self.tableModelFields.newItem()

        if len(self.mdl.getFieldIdList()) > 2:
            self.pushButtonDelete.setEnabled(1)
        else:
            self.pushButtonDelete.setEnabled(0)

        self.browser.configureTree(self.case)


    @pyqtSlot()
    def slotDeleteField(self):
        """
        Delete the Field from the list (one by one).
        """
        row = self.tableViewFields.currentIndex().row()
        if row >= 0 :
            log.debug("slotDeleteProfile -> %s" % row)
            self.tableModelFields.deleteItem(row)

        if len(self.mdl.getFieldIdList()) > 2:
            self.pushButtonDelete.setEnabled(1)
        else:
            self.pushButtonDelete.setEnabled(0)

        self.browser.configureTree(self.case)

#-------------------------------------------------------------------------------
# End
#-------------------------------------------------------------------------------
