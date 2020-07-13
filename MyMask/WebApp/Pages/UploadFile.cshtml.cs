using System.IO;
using System.Net.Mime;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using Microsoft.Extensions.Options;
using WebApp.Classes;

namespace WebApp.Pages
{
    public class UploadFile : PageModel
    {
        public IFormFile FileUpload { get; set; }

        [BindProperty]
        public string BlazorToken { get; set; }

        private BlazorTokenService _tokenService;
        private readonly IOptions<ToolLocations> _toolLocations;

        public UploadFile(BlazorTokenService tokenService, IOptions<ToolLocations> toolLocations)
        {
            _tokenService = tokenService;
            _toolLocations = toolLocations;
        }
        
        public void OnGet()
        {
            
        }

        public async Task<IActionResult> OnPostAsync()
        {
            if (!_tokenService.Tokens.Contains(BlazorToken) ||
                FileUpload.Length > 10485760 ||
                FileUpload.ContentType != MediaTypeNames.Image.Jpeg) return new UnauthorizedResult();
            
            // Make the directory
            string newDirectoryPath = Path.Combine(_toolLocations.Value.TmpStorage, BlazorToken);
            if (IsDirectoryTraversal(newDirectoryPath)) return new UnauthorizedResult();
            Directory.CreateDirectory(newDirectoryPath);
            
            // Save the file
            string newFilePath = Path.Combine(newDirectoryPath, BlazorToken + ".jpg");
            await using FileStream stream = System.IO.File.Create(newFilePath);
            await FileUpload.CopyToAsync(stream);

            _tokenService.Tokens.Remove(BlazorToken);
            return new EmptyResult();
        }
        
        private static bool IsDirectoryTraversal(string path)
        {
            return Path.GetFullPath(path) != path;
        }
    }
}