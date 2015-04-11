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