from django import forms

class ReadOnlyText(forms.TextInput):
  input_type = 'text'


class NameForm(forms.Form):
    Message = forms.CharField(widget=ReadOnlyText, label='message')
