from typing import Dict, Any, List, Union
import yaml
import sys
import re
from pathlib import Path

from PySide2.QtWidgets import (  # type: ignore
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QLabel,
    QFormLayout,
    QTabWidget,
    QWidget,
    QMainWindow,
    QFileDialog,
    QApplication
)


class FocusQLineEdit(QLineEdit):
    """
    FocusQLineEdit subclasses QLineEdit to add the following functionality:

    - Observing focus loss events and, when focus is lost

    - Updating a value in given dictionary and key based on the input
      in this text field.
    """

    def __init__(self, *args, **kwargs):
        """
        The instance attributes for this class are:

        _dictionary: Dict[str, Any]
            The dictionary to be changed when this text field is changed.

        _key_on_dictionary: str
            The key on the dictionary to be changed when this text field
            changes.

        _list_re: re.Pattern
            The regex that matches a string that contains a list, so that
            it does not need to be instantiated each time it is used.

        These attributes default to None. If either one of them is None, no
        attempt will be made to change the dictionary.
        """
        self._dictionary = None
        self._key_on_dictionary = None
        self._list_re = re.compile(r"^\[.*\]$")
        super(FocusQLineEdit, self).__init__(*args, *kwargs)

    def set_dictionary_and_key(self, dictionary: Dict[str, Any], key: str) -> None:
        """
        This method sets the dictionary and key to be modified when the focus
        changes out of this widget

        Parameters
        ----------
        dictionary: Dict[str, Any]
            The dictionary to be changed.

        key: str
            The key whose value is to be changed on the dictionary.
        """
        self._dictionary = dictionary
        self._key_on_dictionary = key

    def focusOutEvent(self, arg__1) -> None:
        """
        Overrides focusOutEvent() in the base class (the base class's method is
        called before anything in this method executes, though).

        The purpose of this override is to observe focus out events. When this
        control looses focus, it updates the underlying specified in the
        instance attributes.
        """
        super(FocusQLineEdit, self).focusOutEvent(arg__1)
        if self._dictionary is not None and self._key_on_dictionary is not None:
            if self.is_list(self.text()):
                value = self.parse_list()
            elif self.is_float(self.text()):
                value = float(self.text())  # type: ignore
            else:
                value = self.text()  # type: ignore
            self._dictionary[self._key_on_dictionary] = value
            print("New level dictionary", self._dictionary)
        else:
            print("Focus lost, but dictionary and key are not set.")

    def parse_list(self) -> List[Union[str, float]]:
        """
        This parses the text in the field to a list of numbers and/or strings.

        Returns
        -------
        List[Union[str, float]]
            A list that contains strings and floats, depending on what could be
            parsed out of the text in the field.
        """
        trimmed_text = self.text()[1:-1]
        trimmed_str_values = [x.strip() for x in trimmed_text.split(",")]
        result: List[Union[str, float]] = []
        for x in trimmed_str_values:
            if self.is_float(x):
                result.append(float(x))
            else:
                result.append(x)
        return result

    @staticmethod
    def is_float(value: Any) -> bool:
        """
        This tests if a value is a float and returns True or False depending
        on the outcome of the test

        Parameters
        ----------
        value: Any
            The value to test.

        Returns
        -------
        bool
            True if the value is a float, False otherwise.
        """
        try:
            float(value)
            return True
        except ValueError:
            return False

    def is_list(self, value: str) -> bool:
        """
        Determines whether a value is a list through a regular expression
        match. The intent is that, if the value is a list, it will be
        parsed as a list.

        Parameters
        ----------
        value: str

        Returns
        -------
        bool
            True if the value appears to be a list, false otherwise
        """
        return self._list_re.match(value) is not None


class FormAndMenuWindow(QMainWindow):
    """
    This class creates a form to edit a dictionary. It nests tabs for different
    levels of the nesting within the dictionaries.

    This automatically builds an interface from the dictionary.
    """

    def __init__(self, parent=None):
        """
        Parameters
        ----------
        dict_to_edit: Dict[str, Any]
            The dictionary to edit in this form.

        output_filename: str
            The filename to write the dictionary when the save button is clicked.

        parent
            The parent as needed by the base class.
        """
        super(FormAndMenuWindow, self).__init__(parent)
        self.analysis_yaml_editor_widget = None
        self.modeling_yaml_widget = None
        self.geometry_yaml_widget = None
        self.geometry_filename_line_edit = None
        self.modeling_filename_line_edit = None
        self.analysis_filename_line_edit = None

    def setup(self) -> None:
        """
        After this class is instantiated, this method should be called to
        lay out the user interface.
        """
        self.setWindowTitle("YAML GUI")
        # self.setup_menu_bar()

        weis_selection_central_widget = self.create_weis_selection_central_widget()
        self.setCentralWidget(weis_selection_central_widget)

    def recursion_ui_setup(self, _dict: Dict[str, Any]) -> QFormLayout:
        """
        This recursive method is where the automatic layout magic happens.
        This method calls itself recursively as it descends down the dictionary
        nesting structure.

        Basically, any given dictionary can consist of scalar and dictionary
        values. At each level of the dictionary, edit fields are placed for
        scalar values and tabbed widgets are placed for the next level of
        nesting.

        Parameters
        ----------
        _dict: Dict[str, Any]
            The dictionary to automatically lay out in to the interface.
        """
        form_level_layout = QFormLayout()
        dict_tabs = QTabWidget()
        display_tabs = False
        for k, v in _dict.items():

            # Recursive call for nested dictionaries.
            if type(v) is dict:
                display_tabs = True
                child_widget = QWidget()
                child_layout = self.recursion_ui_setup(v)
                child_widget.setLayout(child_layout)
                dict_tabs.addTab(child_widget, k)

            # Otherwise just lay out a label and text field.
            else:
                line_edit = FocusQLineEdit(str(v))
                line_edit.set_dictionary_and_key(_dict, k)
                form_level_layout.addRow(QLabel(k), line_edit)

        # If there is a nested dictionary, display it.
        if display_tabs:
            form_level_layout.addRow(dict_tabs)

        # Return the whole layout
        return form_level_layout

    def write_dict_to_yaml(self) -> None:
        """
        This is the event handler for the save button. It simply writes the
        dictionary (which has been continuously updated during focus out
        events) to a YAML file as specified in self.output_filename.
        """
        with open(self.output_filename, "w") as file:
            yaml.dump(self.dict_to_edit, file)

    def create_weis_selection_central_widget(self) -> QWidget:
        """
        Returns
        -------
        QWidget
            The form with buttons on it.
        """
        main_widget = QWidget()

        geometry_section_label = QLabel("Geometry")
        geometry_section_label.setStyleSheet("font-weight: bold;")
        geometry_filename_button = QPushButton("Select geometry YAML...")
        geometry_visualize_button = QPushButton("Visualize geometry")
        self.geometry_filename_line_edit = QLineEdit()
        self.geometry_filename_line_edit.setPlaceholderText(
            "Please select a geometry file."
        )
        self.geometry_filename_line_edit.setReadOnly(True)
        geometry_filename_button.clicked.connect(self.file_picker_geometry)

        modeling_section_label = QLabel("Modeling")
        modeling_section_label.setStyleSheet("font-weight: bold;")
        modeling_filename_button = QPushButton("Select modeling YAML...")
        self.modeling_filename_line_edit = QLineEdit()
        self.modeling_filename_line_edit.setPlaceholderText(
            "Please select a modeling file."
        )
        self.modeling_filename_line_edit.setReadOnly(True)
        modeling_filename_button.clicked.connect(self.file_picker_modeling)

        analysis_section_label = QLabel("Analysis")
        analysis_section_label.setStyleSheet("font-weight: bold;")
        analysis_filename_button = QPushButton("Select analysis YAML...")
        self.analysis_filename_line_edit = QLineEdit()
        self.analysis_filename_line_edit.setPlaceholderText(
            "Please select an analysis file..."
        )
        self.analysis_filename_line_edit.setReadOnly(True)
        analysis_filename_button.clicked.connect(self.file_picker_analysis)

        run_weis_button = QPushButton("Run WEIS")

        self.modeling_yaml_widget = QWidget()
        self.analysis_yaml_editor_widget = QWidget()
        self.geometry_yaml_widget = QWidget()

        geometry_layout = QFormLayout()
        geometry_layout.addRow(geometry_section_label)
        geometry_layout.addRow(
            self.geometry_filename_line_edit, geometry_filename_button
        )
        geometry_layout.addRow(geometry_visualize_button)
        geometry_layout.addRow(self.geometry_yaml_widget)
        geometry_widget = QWidget()
        geometry_widget.setFixedWidth(500)
        geometry_widget.setLayout(geometry_layout)

        modeling_layout = QFormLayout()
        modeling_layout.addRow(modeling_section_label)
        modeling_layout.addRow(
            self.modeling_filename_line_edit, modeling_filename_button
        )
        modeling_layout.addRow(self.modeling_yaml_widget)
        modeling_widget = QWidget()
        modeling_widget.setFixedWidth(500)
        modeling_widget.setLayout(modeling_layout)

        analysis_layout = QFormLayout()
        analysis_layout.addRow(analysis_section_label)
        analysis_layout.addRow(
            self.analysis_filename_line_edit, analysis_filename_button
        )
        analysis_layout.addRow(self.analysis_yaml_editor_widget)
        analysis_widget = QWidget()
        analysis_widget.setFixedWidth(500)
        analysis_widget.setLayout(analysis_layout)

        main_layout = QHBoxLayout()
        main_layout.addWidget(geometry_widget)
        main_layout.addWidget(modeling_widget)
        main_layout.addWidget(analysis_widget)
        main_layout.addWidget(run_weis_button)

        main_widget.setLayout(main_layout)
        return main_widget

    def file_picker_geometry(self):
        """
        Shows the open dialog

        Returns
        -------
        None
            Returns nothing for now.
        """
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setViewMode(QFileDialog.Detail)
        filename, _ = dialog.getOpenFileName(
            None, "Open File", str(Path.home()), "YAML (*.yml *.yaml)"
        )
        self.geometry_filename_line_edit.setText(filename)
        _dict = self.read_yaml_to_dictionary(filename)
        layout = self.recursion_ui_setup(_dict)
        self.geometry_yaml_widget.setLayout(layout)

    def file_picker_modeling(self):
        """
        Shows the open dialog

        Returns
        -------
        None
            Returns nothing for now.
        """
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setViewMode(QFileDialog.Detail)
        filename, _ = dialog.getOpenFileName(
            None, "Open File", str(Path.home()), "YAML (*.yml *.yaml)"
        )
        self.modeling_filename_line_edit.setText(filename)
        _dict = self.read_yaml_to_dictionary(filename)
        layout = self.recursion_ui_setup(_dict)
        self.modeling_yaml_widget.setLayout(layout)

    def file_picker_analysis(self) -> None:
        """
        Shows the open dialog for the analysis YAML

        Returns
        -------
        None
        """
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.setViewMode(QFileDialog.Detail)
        filename, _ = dialog.getOpenFileName(
            None, "Open File", str(Path.home()), "YAML (*.yml *.yaml)"
        )
        self.analysis_filename_line_edit.setText(filename)
        _dict = self.read_yaml_to_dictionary(filename)
        layout = self.recursion_ui_setup(_dict)
        self.analysis_yaml_editor_widget.setLayout(layout)

    @staticmethod
    def read_yaml_to_dictionary(input_filename: str) -> Dict[str, Any]:
        """
        This reads the YAML input which is used to build the user interface.
        """
        with open(input_filename) as file:
            result = yaml.load(file, Loader=yaml.FullLoader)
        return result


def run():
    # Create the Qt Application
    app = QApplication(sys.argv)
    # Create and show the form
    form = FormAndMenuWindow()
    form.setup()
    form.show()
    # Run the main Qt loop
    sys.exit(app.exec_())
