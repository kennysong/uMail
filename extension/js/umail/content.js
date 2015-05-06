// Inject these script tags!
// Actual application code is run from umail.js after injection
var jqueryTag = document.createElement('script');
jqueryTag.src = chrome.extension.getURL('js/gmail-js/jquery-1.10.2.min.js');
(document.head || document.documentElement).appendChild(jqueryTag);

var gmailTag = document.createElement('script');
gmailTag.src = chrome.extension.getURL('js/gmail-js/gmail.js');
(document.head || document.documentElement).appendChild(gmailTag);

var umailTag = document.createElement('script');
umailTag.src = chrome.extension.getURL('js/umail/umail.js');
(document.head || document.documentElement).appendChild(umailTag);

// Hack to expose the extension URL to global scope
var extensionURL = chrome.extension.getURL('');
var uMailURLTag = document.createElement('script');
uMailURLTag.text = 'var uMailExtensionURL = "' + extensionURL + '";';
(document.head || document.documentElement).appendChild(uMailURLTag);

// Listen to messages from the injected umail.js script
window.addEventListener('message', function(event) {
    // Message background.js to send a request to /new_email
    if (event.data.type == 'new_email_request') {
        chrome.runtime.sendMessage({
            method: 'POST',
            action: 'xhttp',
            url: 'http://52.6.28.16/new_email',
            data: event.data.data
        }, function(responseText) {
            // Message umail.js with the /new_email response text
            window.postMessage({type: 'new_email_response', data: responseText}, '*');
        });

    // Message background.js to get the summary variables 
    } else if (event.data.type == 'request_summary_variables') {
        chrome.runtime.sendMessage({ action: 'request_summary_variables' }, function(summary_variables) {
            // Message umail.js with the summary variables
            window.postMessage({type: 'summary_variables_response', data: summary_variables}, '*');
        });
    }
});

// Listen for message from background.js
chrome.runtime.onMessage.addListener(function(request, sender, callback) {
    // Update summary variables in umail.js
    if (request.action == "update_summary_variables") {
        window.postMessage({type: 'update_summary_variables', data: request}, '*');
    }  
});