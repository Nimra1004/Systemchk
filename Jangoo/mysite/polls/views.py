
from django.views.generic.base import TemplateView
from django.shortcuts import render
from django.template import RequestContext
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_protect
# In[1]
from . import predict
from . import content
from . import Generate
from . import Main
from . import nlu
from . import nlg
import json
from django.template import loader




#class Home(TemplateView):

    #template_name = 'Home.html'



from .forms import NameForm
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.csrf import ensure_csrf_cookie
@ensure_csrf_cookie
@csrf_exempt
def get_response(request):
    response = {'status': None}
    if request.method == 'POST':
        #strjson = '{"message": "1"}'
        print(Main.sadia)
        hy = str(Main.sadia)
        strjson = hy
        print('reloaded')
        reply = request.body.decode('utf-8')
        print(reply)

        #data = json.loads(strjson)
        #message = data['message']
        print("done")
        chat_response = strjson
        response['message'] = {'text': chat_response}
        response['status'] = 'ok'
        return HttpResponse(
            json.dumps(response),
            content_type="application/json"
        )

    else:
        print("sadia")
        response['error'] = 'no post data found'

        return HttpResponse(
            json.dumps(response),
            content_type="application/json"
        )
def render_to_response(template_name, context=None, content_type=None, status=None, using=None):
    """
    Returns a HttpResponse whose content is filled with the result of calling
    django.template.loader.render_to_string() with the passed arguments.
    """
    content = loader.render_to_string(template_name, context, using=using)
    return HttpResponse(content, content_type, status)

def home1(request, template_name="home1.html"):
    context = {'title': 'hi'}
    return render_to_response(template_name, context)




