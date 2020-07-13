async function uploadFile(sessionToken) {
    let file = $('#fileUpload')[0].files[0];
    if(file === undefined) return "File is undefined.";
    let fd = new FormData();
    fd.append('FileUpload', file);
    fd.append('BlazorToken', sessionToken);
    fd.append('__RequestVerificationToken', $("input[name='__RequestVerificationToken']").val());
    let response = await fetch("/UploadFile", {
        method: 'POST',
        body: fd
    })

    return response.ok;
}